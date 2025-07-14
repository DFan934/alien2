# ---------------------------------------------------------------------------
# tests/test_staged_exits.py – integration test for M2 (Stop & Staged Exits)
# ---------------------------------------------------------------------------
# This is a *stand-alone* test module.  Drop it under your repo’s ``tests/``
# directory, install pytest-asyncio, and run ``pytest -q tests/test_staged_exits.py``.
# ---------------------------------------------------------------------------

from datetime import datetime
from typing import Any

import pytest

from execution.core.contracts import TradeSignal
from execution.manager import ExecutionManager
from execution.stop_manager import StopManager
from execution.exit_manager import ExitManager
from execution.risk_manager import RiskManager
from execution.latency import latency_monitor
from prediction_engine.ev_engine import EVEngine


# ---------------------------------------------------------------------------
# Dummy EVEngine stub – we only need a .centers attribute for typing
# ---------------------------------------------------------------------------
class DummyPE(EVEngine):
    def __init__(self, *args, **kwargs):
        # override and swallow EVEngine.__init__ signature
        # we never actually use any of its internals in this test
        return

    # satisfy the type‐checker for ExecutionManager.centers
    centers: Any = ...


# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_staged_exit_flow(tmp_path):
    """Full flow: open position → hit TP1 → hit TP2 → position closed."""

    # ---- 0. wiring ---------------------------------------------------------
    exec_mgr = ExecutionManager(
        ev=DummyPE(),
        risk_mgr=RiskManager(account_equity=10_000),
        lat_monitor=latency_monitor,
        config={},
        log_path=tmp_path / "sig.log",
    )

    stop_mgr = StopManager(exec_mgr.store)
    exit_mgr = ExitManager(exec_mgr.store)

    # ---- 1. open position via TradeSignal ----------------------------------
    sig = TradeSignal(
        symbol="TEST",
        side="BUY",
        price=100.0,
        atr=1.0,
        confidence=0.8,
        timestamp=datetime.utcnow(),
        vwap_dist=0.0,
        ema_fast_dist=0.0,
        orderflow_delta=0.0,
        regime="trend",
    )

    await exec_mgr.handle_signal(sig)

    pos = exec_mgr.store.get(sig.id)
    assert pos is not None, "Position not stored"
    qty = pos[3]

    # ---- 2. simulate ticks to +2 % (TP1) and +4 % (TP2) --------------------
    prices = [102.0, 104.0]
    orderflow = [1000.0, 2000.0]

    for px, delta in zip(prices, orderflow):
        # emulate incoming TradeSignal.update with fresh metrics
        stop_mgr.update(
            symbol="TEST",
            side="BUY",
            price=px,
            ema_fast_dist=-0.01,
            vwap_dist=-0.005,
            atr=1.0,
        )
        exit_actions = exit_mgr.on_tick(
            symbol="TEST",
            price=px,
            orderflow_delta=delta,
        )
        # process exit actions through PositionStore effect simulation
        for act in exit_actions:
            if act["type"] == "FINAL":
                exec_mgr.store.close_position(sig.id, act["pnl"])

    # ---- 3. ensure position fully closed -----------------------------------
    assert exec_mgr.store.get(sig.id) is None, "Position should be closed after TP2"

    open_rows = list(exec_mgr.store.list_open())
    assert len(open_rows) == 0, "No open positions expected"

    # ---- 4. check staged exit sizes (50 % then 50 %) ------------------------
    realised = list(
        exec_mgr.store._conn.execute("SELECT pnl FROM realised_pnl").fetchall()  # noqa: WPS437
    )
    assert len(realised) == 2, "Exactly two realised legs expected"

    # CORRECTED LINE: Removed pytest.approx for the inequality check
    assert realised[0][0] < realised[1][0], "Pnl of TP2 should exceed TP1"