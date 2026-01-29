import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest

from execution.manager import ExecutionManager
from execution.brokers.base import MockBrokerAdapter


def _read_parquet_any(path: Path) -> pd.DataFrame:
    """
    Reads either:
      - a single parquet file at `path`
      - a dataset directory at `path` containing *.parquet or part-*.parquet
    Returns empty DF if nothing exists.
    """
    if not path.exists():
        return pd.DataFrame()

    if path.is_file():
        return pd.read_parquet(path)

    # directory dataset
    files = sorted(list(path.glob("*.parquet")) + list(path.glob("part-*.parquet")))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


async def _call_reconcile_once(*, broker: Any, out_dir: Path, store: Any) -> Any:
    """
    Call the Step 11 reconciler one time.

    Supports:
      - class Reconciler(...).<one-shot method>()
      - module-level async function reconcile_once(...)
    """
    import execution.reconciler as rec_mod

    # module-level function support
    if hasattr(rec_mod, "reconcile_once"):
        fn = getattr(rec_mod, "reconcile_once")
        if callable(fn):
            return await fn(broker=broker, store=store, out_dir=out_dir)

    # class-based support
    if hasattr(rec_mod, "Reconciler"):
        r = rec_mod.Reconciler(broker=broker, store=store, out_dir=out_dir)

        # try common one-shot method names
        candidates = [
            "reconcile_once",
            "run_once",
            "tick",
            "step",
            "reconcile",
            "run",
        ]
        for name in candidates:
            meth = getattr(r, name, None)
            if meth is not None and callable(meth):
                return await meth()

        # if none found, show available callable attrs for quick debug
        avail = sorted(
            n for n in dir(r)
            if not n.startswith("_") and callable(getattr(r, n, None))
        )
        raise AssertionError(
            "Found execution.reconciler.Reconciler but no recognized one-shot method. "
            f"Tried {candidates}. Available callables: {avail}"
        )

    raise AssertionError(
        "Step 11 missing API: expected execution/reconciler.py to expose "
        "Reconciler or reconcile_once"
    )



def _set_broker_snapshots(
    broker: Any,
    *,
    positions: List[Dict[str, Any]],
    open_orders: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Make MockBrokerAdapter return broker-truth snapshots.

    If your MockBrokerAdapter already implements these methods, this will just overwrite them.
    If not, we monkeypatch them onto the instance.
    """
    open_orders = open_orders or []

    async def _pos_snap() -> List[Dict[str, Any]]:
        return positions

    async def _orders_snap() -> List[Dict[str, Any]]:
        return open_orders

    setattr(broker, "get_positions_snapshot", _pos_snap)
    setattr(broker, "get_open_orders_snapshot", _orders_snap)


@pytest.mark.asyncio
async def test_step11_reconcile_restart_restores_local_view(tmp_path: Path):
    """
    Acceptance: kill process mid-trade, restart, reconcile restores correct local view.
    This test simulates:
      - a local PositionStore that is wrong/outdated
      - broker snapshot is truth
      - after restart + reconcile_once, local matches broker truth
      - reconciliation_log.parquet exists and has at least 1 row
    """
    out_dir = tmp_path / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- "first run": create manager, attach broker, seed local (WRONG) ----
    mgr1 = ExecutionManager(equity=10_000)
    broker = MockBrokerAdapter()
    await broker.connect()
    mgr1.attach_broker(broker, out_dir=out_dir)

    # Seed local open position (WRONG qty=2)
    # signal_id must be unique inside the SQLite store.
    mgr1.store.add_position(
        signal_id="s1",
        symbol="AAPL",
        side="BUY",
        qty=2,
        entry_px=100.0,
        stop_px=98.0,
        tp_remaining=104.0,
    )

    # Broker truth says qty=1 (e.g., partial fill/partial close happened while we were down)
    _set_broker_snapshots(
        broker,
        positions=[{"symbol": "AAPL", "side": "BUY", "qty": 1, "avg_entry_price": 100.0}],
        open_orders=[],
    )

    # ---- "crash": do NOT gracefully close mgr1; just drop it ----
    # We *do* close broker cleanly so the event loop isn't messy in the test environment.
    await mgr1.aclose()

    # ---- "restart": new manager uses SAME out_dir -> must reuse same positions.db ----
    mgr2 = ExecutionManager(equity=10_000)
    mgr2.attach_broker(broker, out_dir=out_dir)

    try:
        # Reconcile should rewrite local store to match broker snapshot
        await _call_reconcile_once(broker=broker, out_dir=out_dir, store=mgr2.store)

        row = mgr2.store.get_open_symbol("AAPL")
        assert row is not None, "Expected AAPL to exist after reconciliation"

        # Your PositionStore row shape appears to be:
        # (signal_id, symbol, side, qty, entry_px, stop_px, tp_remaining, ts)
        _, sym, side, qty, entry_px, *_ = row
        assert sym == "AAPL"
        assert side == "BUY"
        assert int(qty) == 1
        assert float(entry_px) == 100.0

        # Artifact exists
        log_path = out_dir / "reconciliation_log.parquet"
        df = _read_parquet_any(log_path)
        assert not df.empty, "Expected reconciliation_log.parquet to have at least one row"

        # Basic expected columns (adjust if your schema differs)
        expected_cols = {"ts_utc", "symbol", "action"}
        missing = expected_cols - set(df.columns)
        assert not missing, f"reconciliation_log.parquet missing columns: {missing}"

    finally:
        await mgr2.aclose()
        await broker.close()


@pytest.mark.asyncio
async def test_step11_reconcile_is_idempotent(tmp_path: Path):
    """
    Running reconcile twice in a row should not keep producing 'changes'.
    We verify either:
      - no new rows on second pass, OR
      - second pass rows have action='NOOP' only
    """
    out_dir = tmp_path / "run"
    out_dir.mkdir(parents=True, exist_ok=True)

    mgr = ExecutionManager(equity=10_000)
    broker = MockBrokerAdapter()
    await broker.connect()
    mgr.attach_broker(broker, out_dir=out_dir)

    try:
        # Local is wrong qty=2
        mgr.store.add_position(
            signal_id="s1",
            symbol="AAPL",
            side="BUY",
            qty=2,
            entry_px=100.0,
            stop_px=98.0,
            tp_remaining=104.0,
        )

        # Broker truth qty=2 (so reconcile should do nothing) OR
        # change to qty=1 to force one change, then second pass should be no-op.
        _set_broker_snapshots(
            broker,
            positions=[{"symbol": "AAPL", "side": "BUY", "qty": 1, "avg_entry_price": 100.0}],
            open_orders=[],
        )

        log_path = out_dir / "reconciliation_log.parquet"

        # First reconcile: should produce at least one action row
        await _call_reconcile_once(broker=broker, out_dir=out_dir, store=mgr.store)
        df1 = _read_parquet_any(log_path)
        assert not df1.empty, "Expected reconciliation log to exist after first reconcile"
        n1 = len(df1)

        # Second reconcile: should not keep producing changes
        await _call_reconcile_once(broker=broker, out_dir=out_dir, store=mgr.store)
        df2 = _read_parquet_any(log_path)
        n2 = len(df2)

        if n2 > n1:
            # If you log every pass, enforce second pass is NOOP-only rows
            new_rows = df2.iloc[n1:]
            if "action" in new_rows.columns:
                assert set(new_rows["action"].astype(str).str.upper().unique()) <= {"NOOP"}, (
                    f"Expected idempotent reconcile; got non-NOOP actions on second pass: "
                    f"{new_rows['action'].unique()}"
                )
            else:
                # If no action column, require strictly no additional rows
                pytest.fail("Reconcile wrote additional rows but log has no 'action' column to verify NOOP")
        else:
            # Preferred: no new rows on second pass
            assert n2 == n1

        # Also confirm local is now correct
        row = mgr.store.get_open_symbol("AAPL")
        assert row is not None
        assert int(row[3]) == 1

    finally:
        await mgr.aclose()
        await broker.close()
