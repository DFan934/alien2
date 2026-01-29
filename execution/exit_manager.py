# execution/exit_manager.py

from __future__ import annotations
from typing import List, Dict
from execution.position_store import PositionStore

class ExitManager:
    """Handles staged exits & full closures."""

    TP1_PCT = 0.02
    TP2_PCT = 0.04

    def __init__(self, store: PositionStore):
        self.store = store

    # ----------------------------------------------------------------
    def on_tick(
        self,
        symbol: str,
        price: float,
        orderflow_delta: float,
        reversal: bool = False,
        profile: dict | None = None,
    ) -> List[Dict]:
        """
        Returns a list of exit actions when price or reversal conditions
        hit tiered profit targets.  Only the partial (TP1) exit is
        persisted here; the final exit is left to the caller.
        """
        tp_list = profile.get("tp_pct") if profile else None
        tp1_pct, tp2_pct = (
            tp_list if tp_list and len(tp_list) == 2 else
            (self.TP1_PCT, self.TP2_PCT)
        )

        row = self.store.get_open_symbol(symbol)
        if row is None:
            return []
        #signal_id, _sym, side, qty, entry_px, _stop_px, tp_remaining, _ = row
        signal_id, _sym, side, qty, entry_px, stop_px, tp_remaining, _ = row

        actions: List[Dict] = []

        def calc_pnl(exit_qty: int) -> float:
            if side == "BUY":
                return exit_qty * (price - entry_px)
            else:
                return exit_qty * (entry_px - price)

        tp1 = entry_px * (1 + tp1_pct if side == "BUY" else 1 - tp1_pct)
        tp2 = entry_px * (1 + tp2_pct if side == "BUY" else 1 - tp2_pct)

        # --- stop-loss: close everything if stop breached ------------
        if stop_px is not None and (
            (side == "BUY" and price <= stop_px) or
            (side == "SELL" and price >= stop_px)
        ):
            actions.append({
                "type": "STOP",
                "reason": "stop",
                "qty": qty,
                "pnl": calc_pnl(qty),
            })
            return actions


        # --- first tier: close half at TP1 -------------------------
        '''if tp_remaining == tp2 and (
            (side == "BUY" and price >= tp1) or
            (side == "SELL" and price <= tp1)
        ):'''
        # TP1 should fire only if we haven't already taken it.
        # Using float equality (tp_remaining == tp2) is fragile, so treat
        # "not yet TP1" as tp_remaining being closer to TP2 than to TP1.
        # --- first tier: close half at TP1 -------------------------
        # Don't use float equality as a state machine.
        # Treat "TP1 not yet taken" as tp_remaining not already being set to tp1.
        tp1_not_taken = (tp_remaining is None) or (abs(tp_remaining - tp1) > 1e-9)

        if tp1_not_taken and (
                (side == "BUY" and price >= tp1) or
                (side == "SELL" and price <= tp1)
        ):
            half = qty // 2 or 1
            pnl1 = calc_pnl(half)

            # THIS MANUAL INSERT WAS THE SOURCE OF THE BUG AND HAS BEEN REMOVED.
            # self.store._conn.execute(
            #     "INSERT INTO realised_pnl (signal_id, pnl, closed_at) "
            #     "VALUES (?, ?, strftime('%s','now'))",
            #     (signal_id, pnl1),
            # )

            # The reduce_position method is now solely responsible for handling
            # the partial close, including logging the PnL.
            self.store.reduce_position(signal_id, half)
            self.store.update_tp_remaining(signal_id, tp1)

            actions.append({
                "type": "TP",
                "pct": 0.5,
                "qty": half,
                "pnl": pnl1,
            })

            qty -= half
            tp_remaining = tp1

        # --- final tier / reversal: close any remainder -------------
        hit_tp2 = (side == "BUY" and price >= tp2) or (side == "SELL" and price <= tp2)
        reversal_cond = (
            reversal or
            (side == "BUY" and orderflow_delta < 0) or
            (side == "SELL" and orderflow_delta > 0)
        )
        if qty > 0 and (hit_tp2 or reversal_cond):
            pnl2 = calc_pnl(qty)

            actions.append({
                "type": "FINAL",
                "reason": "tp2" if hit_tp2 else "reversal",
                "qty": qty,
                "pnl": pnl2,
            })

        return actions