# -------------------------------------------------------------------
# execution/position_store.py   **REPLACE ENTIRE FILE**
# -------------------------------------------------------------------
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Union, List, Tuple

from execution.core.contracts import SafetyAction


class PositionStore:
    """Tracks open positions, realised PnL, and safety actions."""

    def __init__(self, db_path: Union[str, Path, None] = None) -> None:
        # Default to in-memory unless explicitly given a path
        path = ":memory:" if db_path is None else str(db_path)

        # For file DBs, ensure parent exists
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(path)
        self._init_schema()

    # ----------------------------------------------------------------
    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS positions (
                signal_id    TEXT PRIMARY KEY,
                symbol       TEXT,
                side         TEXT,
                qty          INTEGER,
                entry_px     REAL,
                stop_px      REAL,
                tp_remaining REAL,
                opened_at    REAL
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS realised_pnl (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id TEXT,
                pnl       REAL,
                closed_at REAL
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS safety_log (
                id     INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT,
                reason TEXT,
                ts     REAL
            );
            """
        )
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.commit()
        finally:
            self._conn.close()

    # ---------------- position helpers --------------------------------
    def add_position(
        self,
        signal_id: str,
        symbol: str,
        side: str,
        qty: int,
        entry_px: float,
        stop_px: float,
        tp_remaining: float,
    ) -> None:
        self._conn.execute(
            "INSERT INTO positions VALUES (?,?,?,?,?,?,?, strftime('%s','now'))",
            (signal_id, symbol, side, qty, entry_px, stop_px, tp_remaining),
        )
        self._conn.commit()

    def update_tp_remaining(self, signal_id: str, tp: float) -> None:
        self._conn.execute(
            "UPDATE positions SET tp_remaining = ? WHERE signal_id = ?",
            (tp, signal_id),
        )
        self._conn.commit()

    def reduce_position(self, signal_id: str, qty_to_close: int) -> None:
        cur = self._conn.execute("SELECT qty FROM positions WHERE signal_id = ?", (signal_id,))
        row = cur.fetchone()
        if not row:
            return
        old_qty = int(row[0])

        self._conn.execute(
            "INSERT INTO realised_pnl (signal_id, pnl, closed_at) VALUES (?, ?, strftime('%s','now'))",
            (signal_id, qty_to_close),
        )

        qty_left = max(old_qty - int(qty_to_close), 0)
        if qty_left == 0:
            self._conn.execute("DELETE FROM positions WHERE signal_id = ?", (signal_id,))
        else:
            self._conn.execute("UPDATE positions SET qty = ? WHERE signal_id = ?", (qty_left, signal_id))

        self._conn.commit()

    def update_stop(self, symbol: str, new_stop: float) -> None:
        self._conn.execute("UPDATE positions SET stop_px = ? WHERE symbol = ?", (new_stop, symbol))
        self._conn.commit()

    def close_all(self, signal_id: str, exit_px: float) -> None:
        cur = self._conn.execute("SELECT qty, entry_px FROM positions WHERE signal_id = ?", (signal_id,))
        row = cur.fetchone()
        if not row:
            return
        qty, entry_px = float(row[0]), float(row[1])
        pnl = (exit_px - entry_px) * qty
        self._conn.execute(
            "INSERT INTO realised_pnl (signal_id, pnl, closed_at) VALUES (?, ?, strftime('%s','now'))",
            (signal_id, pnl),
        )
        self._conn.execute("DELETE FROM positions WHERE signal_id = ?", (signal_id,))
        self._conn.commit()

    def get_open_symbol(self, symbol: str) -> Optional[tuple]:
        sym = symbol.upper()
        cur = self._conn.execute(
            """
            SELECT signal_id, symbol, side, qty, entry_px, stop_px, tp_remaining, opened_at
            FROM positions
            WHERE symbol = ? AND qty > 0
            ORDER BY
                CASE WHEN signal_id LIKE 'BRK:%' THEN 0 ELSE 1 END,
                opened_at DESC
            LIMIT 1
            """,
            (sym,),
        )
        return cur.fetchone()

    def get(self, signal_id: str) -> Optional[tuple]:
        cur = self._conn.execute(
            "SELECT signal_id, symbol, side, qty, entry_px, stop_px, tp_remaining, opened_at "
            "FROM positions WHERE signal_id = ?",
            (signal_id,),
        )
        return cur.fetchone()

    # ---------------- safety log --------------------------------------
    def add_safety(self, action: SafetyAction) -> None:
        self._conn.execute(
            "INSERT INTO safety_log (action, reason, ts) VALUES (?,?, strftime('%s','now'))",
            (action.action, action.reason),
        )
        self._conn.commit()

    def close_position(self, signal_id: str, pnl: float) -> None:
        self._conn.execute(
            "INSERT INTO realised_pnl (signal_id, pnl, closed_at) VALUES (?, ?, strftime('%s','now'))",
            (signal_id, pnl),
        )
        self._conn.execute("DELETE FROM positions WHERE signal_id = ?", (signal_id,))
        self._conn.commit()

    def list_open_positions(self) -> List[Tuple]:
        """
        Returns rows shaped like:
          (signal_id, symbol, side, qty, entry_px, stop_px, tp_remaining, opened_at)
        """
        cur = self._conn.execute(
            "SELECT signal_id, symbol, side, qty, entry_px, stop_px, tp_remaining, opened_at "
            "FROM positions"
        )
        return list(cur.fetchall())

    def upsert_position_from_broker(
            self,
            *,
            symbol: str,
            side: str,
            qty: int,
            entry_px: float,
            stop_px: Optional[float],
            tp_remaining: Optional[float],
            broker_position_id: Optional[str] = None,
    ) -> None:
        """
        Ensures local has a position for this symbol that matches broker truth.
        Uses a deterministic signal_id namespace so restart recovery is idempotent.
        """
        sym = symbol.upper()
        signal_id = f"BRK:{sym}"  # deterministic idempotent id for broker-derived positions

        # If you have UNIQUE(signal_id) already, this is safe.
        # We do a delete+insert to avoid requiring SQLITE UPSERT syntax compatibility.
        self._conn.execute("DELETE FROM positions WHERE signal_id = ?", (signal_id,))
        self._conn.execute(
            "INSERT INTO positions VALUES (?,?,?,?,?,?,?, strftime('%s','now'))",
            (signal_id, sym, side, int(qty), float(entry_px), stop_px, tp_remaining),
        )
        self._conn.commit()

    def close_local_symbol(self, symbol: str, *, closed_at_utc: datetime) -> None:
        """
        Broker says no position exists -> local must remove.
        If you later add a closed_positions table, you can archive here instead.
        """
        sym = symbol.upper()
        self._conn.execute("DELETE FROM positions WHERE symbol = ?", (sym,))
        self._conn.commit()

    def list_open(self) -> Iterable[tuple]:
        return self._conn.execute(
            "SELECT signal_id, symbol, side, qty, entry_px, stop_px, tp_remaining, opened_at FROM positions"
        ).fetchall()
