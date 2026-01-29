# execution/reconciler.py
from __future__ import annotations

import asyncio
import contextlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class ReconcileStats:
    changed: int
    noop: int
    wrote_rows: int


class Reconciler:
    """
    Step 11: reconciliation loop + restart recovery.

    Contract (as used by tests):
      - class Reconciler(...).reconcile_once()  (or run_once/step/etc)
      - writes out_dir/reconciliation_log.parquet with at least columns:
          ts_utc, symbol, action
      - applies broker snapshot truth to local PositionStore such that after restart
        the local view matches broker truth.
    """

    def __init__(self, *, broker: Any, store: Any, out_dir: Path, interval_s: float = 5.0) -> None:
        self.broker = broker
        self.store = store
        self.out_dir = Path(out_dir)
        self.interval_s = float(interval_s)
        self._task: Optional[asyncio.Task] = None

        # We write a "dataset directory" at out_dir/reconciliation_log.parquet/
        # (your tests can read both file or dataset directory)
        self._log_dir = self.out_dir / "reconciliation_log.parquet"
        self._part_idx = 0

    # ---------- public API ----------

    async def reconcile_once(self) -> ReconcileStats:
        """
        One-shot reconcile:
          - read broker snapshot
          - compare to local store
          - apply changes to local store
          - log ONLY when changes happen (idempotent second pass => no new rows)
        """
        ts = datetime.now(timezone.utc)

        broker_positions = await self._safe_broker_positions()
        # open orders not required for your current tests, but you can extend later
        # broker_orders = await self._safe_broker_open_orders()

        # Normalize broker positions by symbol
        bmap: Dict[str, Dict[str, Any]] = {}
        for p in broker_positions:
            sym = str(p.get("symbol", "")).upper()
            if not sym:
                continue
            qty = int(p.get("qty") or 0)
            side = str(p.get("side") or "BUY").upper()
            entry_px = float(p.get("avg_entry_price") or p.get("entry_px") or 0.0)
            bmap[sym] = {"symbol": sym, "qty": qty, "side": side, "entry_px": entry_px}

        changed = 0
        noop = 0
        rows_to_log: List[Dict[str, Any]] = []

        # For every broker-truth position: ensure local matches.
        for sym, bp in bmap.items():
            local = self.store.get_open_symbol(sym)

            if bp["qty"] <= 0:
                # broker has no position => ensure local is cleared
                if local is not None:
                    self._store_clear_symbol(sym)
                    changed += 1
                    rows_to_log.append(
                        {"ts_utc": ts, "symbol": sym, "action": "CLEAR", "local_qty": int(local[3]), "broker_qty": 0}
                    )
                else:
                    noop += 1
                continue

            # broker has a position
            if local is None:
                # create local stub
                self._store_upsert_symbol(
                    sym=sym,
                    side=bp["side"],
                    qty=bp["qty"],
                    entry_px=bp["entry_px"],
                )
                changed += 1
                rows_to_log.append(
                    {"ts_utc": ts, "symbol": sym, "action": "INSERT", "local_qty": 0, "broker_qty": int(bp["qty"])}
                )
                continue

            # local exists; compare qty/side/entry_px
            local_qty = int(local[3])
            local_side = str(local[2]).upper()
            local_entry = float(local[4])

            if local_qty == int(bp["qty"]) and local_side == bp["side"] and abs(local_entry - bp["entry_px"]) < 1e-9:
                noop += 1
                continue

            self._store_upsert_symbol(
                sym=sym,
                side=bp["side"],
                qty=bp["qty"],
                entry_px=bp["entry_px"],
            )
            changed += 1
            rows_to_log.append(
                {
                    "ts_utc": ts,
                    "symbol": sym,
                    "action": "UPDATE",
                    "local_qty": local_qty,
                    "broker_qty": int(bp["qty"]),
                    "local_side": local_side,
                    "broker_side": bp["side"],
                    "local_entry_px": float(local_entry),
                    "broker_entry_px": float(bp["entry_px"]),
                }
            )

        # Optional: if local has symbols broker DOESN’T have, clear them.
        # (This helps restart recovery if broker flattened while you were down.)
        # We do this best-effort: only if PositionStore exposes a list method or sqlite path.
        for sym in self._iter_local_symbols_best_effort():
            if sym not in bmap:
                local = self.store.get_open_symbol(sym)
                if local is not None:
                    self._store_clear_symbol(sym)
                    changed += 1
                    rows_to_log.append(
                        {"ts_utc": ts, "symbol": sym, "action": "CLEAR", "local_qty": int(local[3]), "broker_qty": 0}
                    )

        wrote = 0
        if rows_to_log:
            self._append_log_rows(rows_to_log)
            wrote = len(rows_to_log)

        return ReconcileStats(changed=changed, noop=noop, wrote_rows=wrote)

    # Alias names so your helper’s candidate list always finds something
    async def run_once(self) -> ReconcileStats:
        return await self.reconcile_once()

    async def step(self) -> ReconcileStats:
        return await self.reconcile_once()

    async def run_loop(self) -> None:
        while True:
            await self.reconcile_once()
            await asyncio.sleep(self.interval_s)

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.run_loop())

    async def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None

    # ---------- broker snapshot helpers ----------

    async def _safe_broker_positions(self) -> List[Dict[str, Any]]:
        if hasattr(self.broker, "get_positions_snapshot"):
            return await self.broker.get_positions_snapshot()
        return []

    async def _safe_broker_open_orders(self) -> List[Dict[str, Any]]:
        if hasattr(self.broker, "get_open_orders_snapshot"):
            return await self.broker.get_open_orders_snapshot()
        return []

    # ---------- logging ----------

    def _append_log_rows(self, rows: List[Dict[str, Any]]) -> None:
        """
        Writes an append-only parquet dataset directory:
          out_dir/reconciliation_log.parquet/part-000001.parquet, ...
        """
        self._log_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame.from_records(rows)
        if df.empty:
            return
        self._part_idx += 1
        part = self._log_dir / f"part-{self._part_idx:06d}.parquet"
        df.to_parquet(part, index=False)

    # ---------- store mutation (best-effort / robust) ----------

    def _store_upsert_symbol(self, *, sym: str, side: str, qty: int, entry_px: float) -> None:
        """
        Update/insert local position for `sym` to match broker truth.
        Prefer PositionStore native APIs; fall back only if truly needed.
        """
        sym = sym.upper()

        # 1) Preferred: your PositionStore has a broker-truth upsert API
        '''fn = getattr(self.store, "upsert_position_from_broker", None)
        if callable(fn):
            fn(
                symbol=sym,
                side=side,
                qty=int(qty),
                entry_px=float(entry_px),
                stop_px=None,  # reconciler doesn't know stops from snapshot
                tp_remaining=None,  # reconciler doesn't know tp from snapshot
                broker_position_id=None,
            )
            return'''

        fn = getattr(self.store, "upsert_position_from_broker", None)
        if callable(fn):
            # IMPORTANT: remove any stale local rows for this symbol (e.g. "s1")
            # so the broker-truth row is the only one left.
            clr = getattr(self.store, "close_local_symbol", None)
            if callable(clr):
                clr(sym, closed_at_utc=datetime.now(timezone.utc))

            fn(
                symbol=sym,
                side=side,
                qty=int(qty),
                entry_px=float(entry_px),
                stop_px=None,  # reconciler doesn't know stops from snapshot
                tp_remaining=None,  # reconciler doesn't know tp from snapshot
                broker_position_id=None,
            )
            return

        # 2) Next-best: look for generic update APIs (if present in other store versions)
        for name in ("update_position", "update_open_position", "set_position", "set_open_position"):
            fn2 = getattr(self.store, name, None)
            if callable(fn2):
                fn2(sym, side=side, qty=int(qty), entry_px=float(entry_px))
                return

        # 3) Last resort: keep your old sqlite-introspection fallback (if you want)
        ok = self._sqlite_upsert_by_symbol(sym=sym, side=side, qty=int(qty), entry_px=float(entry_px))
        if not ok:
            raise RuntimeError(
                f"Reconcile failed to upsert local position for {sym}. "
                "Could not locate/update PositionStore backing DB."
            )

    def _store_clear_symbol(self, sym: str) -> None:
        """
        Ensure local store has no open position for sym.
        Best-effort: try a store method; else sqlite delete.
        """
        # If your store has a close/delete method, use it
        for name in ("close_local_symbol", "close_symbol", "delete_symbol", "remove_symbol", "clear_symbol"):
            fn = getattr(self.store, name, None)
            if callable(fn):
                fn(sym)
                return

        # sqlite fallback
        self._sqlite_delete_symbol(sym)

    def _sqlite_update_symbol(self, *, sym: str, side: str, qty: int, entry_px: float) -> None:
        db_path = self._get_store_db_path()
        if db_path is None:
            return

        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            # find a row for this symbol
            cur = conn.execute(
                "SELECT signal_id FROM positions WHERE UPPER(symbol)=UPPER(?) LIMIT 1",
                (sym,),
            )
            got = cur.fetchone()
            if got is None:
                # insert stub
                signal_id = f"recon:{sym}"
                conn.execute(
                    """
                    INSERT INTO positions (signal_id, symbol, side, qty, entry_px, stop_px, tp_remaining, ts_utc)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        signal_id,
                        sym,
                        side,
                        float(qty),
                        float(entry_px),
                        None,
                        None,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
            else:
                signal_id = got[0]
                conn.execute(
                    """
                    UPDATE positions
                    SET side=?, qty=?, entry_px=?
                    WHERE signal_id=?
                    """,
                    (side, float(qty), float(entry_px), signal_id),
                )
            conn.commit()

    def _sqlite_delete_symbol(self, sym: str) -> None:
        db_path = self._get_store_db_path()
        if db_path is None:
            return
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("DELETE FROM positions WHERE UPPER(symbol)=UPPER(?)", (sym,))
            conn.commit()

    def _get_store_db_path(self) -> Optional[Path]:
        """
        Aggressively search for the sqlite file path inside PositionStore.
        Handles many possible attribute names and types.
        """
        # Common direct attribute names
        for attr in ("db_path", "_db_path", "path", "_path", "db", "_db"):
            p = getattr(self.store, attr, None)
            if p:
                try:
                    pp = Path(p)
                    if pp.suffix.lower() == ".db" or pp.name.endswith(".db"):
                        return pp
                except Exception:
                    pass

        # Look through __dict__ for anything path-like ending in .db
        d = getattr(self.store, "__dict__", {}) or {}
        for _, v in d.items():
            try:
                pp = Path(v)
                if pp.suffix.lower() == ".db" or pp.name.endswith(".db"):
                    return pp
            except Exception:
                continue

        return None

    def _sqlite_find_positions_table(self, conn: sqlite3.Connection) -> Optional[Tuple[str, Dict[str, str]]]:
        """
        Find the table that looks like the positions table, and map canonical fields
        to actual column names.

        Returns: (table_name, colmap) where colmap maps:
          "signal_id", "symbol", "side", "qty", "entry_px" -> actual col names
        """
        # Candidate tables
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        if not tables:
            return None

        def cols_for(t: str) -> List[str]:
            rows = conn.execute(f"PRAGMA table_info({t})").fetchall()
            return [str(r[1]) for r in rows]  # (cid, name, type, ...)

        # canonical -> possible column names
        want = {
            "signal_id": ["signal_id", "id", "sig_id"],
            "symbol": ["symbol", "sym", "ticker"],
            "side": ["side", "direction"],
            "qty": ["qty", "quantity", "size", "position_qty", "shares"],
            "entry_px": ["entry_px", "avg_entry_price", "avg_entry_px", "entry_price", "avg_price", "price"],
        }

        for t in tables:
            cols = cols_for(t)
            lower = {c.lower(): c for c in cols}

            colmap: Dict[str, str] = {}
            for canon, options in want.items():
                found = None
                for opt in options:
                    if opt.lower() in lower:
                        found = lower[opt.lower()]
                        break
                if found is None:
                    # require signal_id, symbol, qty at minimum; side/entry_px best-effort
                    if canon in ("signal_id", "symbol", "qty"):
                        colmap = {}
                        break
                else:
                    colmap[canon] = found

            if colmap.get("signal_id") and colmap.get("symbol") and colmap.get("qty"):
                # side/entry_px may be absent; we'll update only what exists
                return (t, colmap)

        return None

    def _sqlite_upsert_by_symbol(self, *, sym: str, side: str, qty: int, entry_px: float) -> bool:
        """
        Update (or insert) a local position row keyed by symbol.
        Returns True if it succeeded, False if no DB / no matching table.
        """
        db_path = self._get_store_db_path()
        if db_path is None or not db_path.exists():
            return False

        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            found = self._sqlite_find_positions_table(conn)
            if found is None:
                return False

            table, m = found
            c_sig = m["signal_id"]
            c_sym = m["symbol"]
            c_qty = m["qty"]
            c_side = m.get("side")
            c_entry = m.get("entry_px")

            # Does row exist?
            cur = conn.execute(
                f"SELECT {c_sig} FROM {table} WHERE UPPER({c_sym})=UPPER(?) LIMIT 1",
                (sym,),
            )
            row = cur.fetchone()

            if row is None:
                # Insert minimal row using whatever cols exist.
                signal_id = f"recon:{sym}"

                cols = [c_sig, c_sym, c_qty]
                vals = [signal_id, sym, float(qty)]

                if c_side:
                    cols.append(c_side)
                    vals.append(side)
                if c_entry:
                    cols.append(c_entry)
                    vals.append(float(entry_px))

                placeholders = ",".join(["?"] * len(cols))
                collist = ",".join(cols)
                conn.execute(
                    f"INSERT INTO {table} ({collist}) VALUES ({placeholders})",
                    tuple(vals),
                )
            else:
                sig = row[0]

                sets = [f"{c_qty}=?"]
                vals2: List[Any] = [float(qty)]

                if c_side:
                    sets.append(f"{c_side}=?")
                    vals2.append(side)
                if c_entry:
                    sets.append(f"{c_entry}=?")
                    vals2.append(float(entry_px))

                vals2.append(sig)

                conn.execute(
                    f"UPDATE {table} SET {', '.join(sets)} WHERE {c_sig}=?",
                    tuple(vals2),
                )

            conn.commit()
            return True

    def _sqlite_delete_symbol(self, sym: str) -> None:
        db_path = self._get_store_db_path()
        if db_path is None or not db_path.exists():
            return
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            found = self._sqlite_find_positions_table(conn)
            if found is None:
                return
            table, m = found
            c_sym = m["symbol"]
            conn.execute(f"DELETE FROM {table} WHERE UPPER({c_sym})=UPPER(?)", (sym,))
            conn.commit()

    def _iter_local_symbols_best_effort(self) -> List[str]:
        db_path = self._get_store_db_path()
        if db_path is None or not db_path.exists():
            return []
        try:
            with sqlite3.connect(str(db_path)) as conn:
                found = self._sqlite_find_positions_table(conn)
                if found is None:
                    return []
                table, m = found
                c_sym = m["symbol"]
                cur = conn.execute(f"SELECT DISTINCT UPPER({c_sym}) FROM {table}")
                return [str(r[0]).upper() for r in cur.fetchall() if r and r[0]]
        except Exception:
            return []

# Optional module-level helper if you ever want it
async def reconcile_once(*, broker: Any, store: Any, out_dir: Path) -> ReconcileStats:
    r = Reconciler(broker=broker, store=store, out_dir=Path(out_dir))
    return await r.reconcile_once()
