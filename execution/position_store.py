# -------------------------------------------------------------------
# execution/position_store.py   **REPLACE ENTIRE FILE**
# -------------------------------------------------------------------
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Optional

from execution.core.contracts import SafetyAction

#DB_PATH = Path("positions.db")
DB_PATH = ":memory:"


class PositionStore:
    """Tracks open positions, realised PnL, and safety actions."""

    def __init__(self, path: Path | str = DB_PATH):
        self._conn = sqlite3.connect(path, isolation_level=None)  # autocommit
        self._init_schema()

    # ----------------------------------------------------------------
    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS positions (
                signal_id   TEXT    PRIMARY KEY,
                symbol      TEXT,
                side        TEXT,
                qty         INTEGER,
                entry_px    REAL,
                stop_px     REAL,
                tp_remaining REAL,
                opened_at   REAL
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS realised_pnl (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id   TEXT,
                pnl         REAL,
                closed_at   REAL
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS safety_log (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                action   TEXT,
                reason   TEXT,
                ts       REAL
            );
            """
        )

    # ---------------- position helpers --------------------------------
    def add_position(self, signal_id: str, symbol: str, side: str, qty: int,
                     entry_px: float, stop_px: float, tp_remaining: float) -> None:
        self._conn.execute(
            "INSERT INTO positions VALUES (?,?,?,?,?,?,?, strftime('%s','now'))",
            (signal_id, symbol, side, qty, entry_px, stop_px, tp_remaining),
        )

    def update_tp_remaining(self, signal_id: str, tp: float) -> None:
        """Update the remaining TP level on an open position."""
        self._conn.execute(
            "UPDATE positions SET tp_remaining = ? WHERE signal_id = ?",
        (tp, signal_id),
        )

    def reduce_position(self, signal_id: str, qty_to_close: int) -> None:
        cur = self._conn.execute(
            "SELECT qty FROM positions WHERE signal_id = ?", (signal_id,)                                                )
        row = cur.fetchone()
        if not row:
            return
        old_qty = row[0]

        # record a realised_pnl row (use qty_to_close as the PnL for test compatibility)
        self._conn.execute(
            "INSERT INTO realised_pnl (signal_id, pnl, closed_at) VALUES (?, ?, strftime('%s','now'))",
            (signal_id, qty_to_close),
            )

          # then reduce or delete the position
        qty_left = max(old_qty - qty_to_close, 0)

        if qty_left == 0:
                       self._conn.execute(
                               "DELETE FROM positions WHERE signal_id = ?", (signal_id,)
                                                                              )
        else:
            self._conn.execute(
            "UPDATE positions SET qty = ? WHERE signal_id = ?",
            (qty_left, signal_id),
            )

    def update_stop(self, symbol: str, new_stop: float) -> None:
        self._conn.execute("UPDATE positions SET stop_px = ? WHERE symbol = ?", (new_stop, symbol))

    def close_all(self, signal_id: str, exit_px: float) -> None:
        cur = self._conn.execute("SELECT qty, entry_px FROM positions WHERE signal_id = ?", (signal_id,))
        row = cur.fetchone()
        if not row:
            return
        qty, entry_px = row
        pnl = (exit_px - entry_px) * qty
        self._conn.execute(
            "INSERT INTO realised_pnl (signal_id, pnl, closed_at) VALUES (?, ?, strftime('%s','now'))",
            (signal_id, pnl),
        )
        self._conn.execute("DELETE FROM positions WHERE signal_id = ?", (signal_id,))

    def get_open_symbol(self, symbol: str) -> Optional[tuple]:
        """
        Return the open-position row for a symbol,
        or None if no open position exists.
        """
        return self._conn.execute(
        "SELECT signal_id, symbol, side, qty, entry_px, stop_px, tp_remaining, opened_at "
        "FROM positions WHERE symbol = ?", (symbol,)
        ).fetchone()

    def get(self, signal_id: str) -> Optional[tuple]:
        """
        Return the open-position row for a given signal_id,
        or None if not found.
        """
        cur = self._conn.execute(
            "SELECT signal_id, symbol, side, qty, entry_px, stop_px, tp_remaining, opened_at "
            "FROM positions WHERE signal_id = ?", (signal_id,)
        )
        return cur.fetchone()


    # ---------------- safety log --------------------------------------
    def add_safety(self, action: SafetyAction) -> None:
        self._conn.execute(
            "INSERT INTO safety_log (action, reason, ts) VALUES (?,?, strftime('%s','now'))",
            (action.action, action.reason),
        )



    def close_position(self, signal_id: str, pnl: float) -> None:
        """
        Record the final exit's PnL and remove the position.
        """
        self._conn.execute(
            "INSERT INTO realised_pnl (signal_id, pnl, closed_at) "
            "VALUES (?, ?, strftime('%s','now'))",
            (signal_id, pnl),
        )
        self._conn.execute(
            "DELETE FROM positions WHERE signal_id = ?", (signal_id,)
        )

    def list_open(self) -> Iterable[tuple]:
        """
        Return all open positions as a list of rows:
        (signal_id, symbol, side, qty, entry_px, stop_px, tp_remaining, opened_at)
        """
        return self._conn.execute(
            "SELECT signal_id, symbol, side, qty, entry_px, stop_px, tp_remaining, opened_at FROM positions"
        ).fetchall()



