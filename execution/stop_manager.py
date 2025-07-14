# execution/stop_manager.py
# ----------------------------------------------------------------
from __future__ import annotations

from typing import Dict, Optional

from execution.position_store import PositionStore


class StopManager:
    """Per-symbol trailing / widening stop logic."""

    def __init__(self, store: PositionStore, atr_mult: float = 1.5):
        self.store = store
        self.atr_mult = atr_mult
        self._entry_atr: Dict[str, float] = {}

    def register_position(self, symbol: str, entry_atr: float):
        self._entry_atr[symbol] = entry_atr

    def update(
        self,
        symbol: str,
        side: str,
        price: float,
        ema_fast_dist: float,
        vwap_dist: float,
        atr: float,
        profile: dict[str, float] | None = None,
    ) -> Optional[float]:
        """
        Optionally tighten or widen the stop based on EMA/VWAP reversion and ATR changes,
        with the ability to override the ATR multiplier via `profile["atr_mult"]`.
        """
        # fetch current position row
        row = self.store.get_open_symbol(symbol)
        if row is None:
            return None

        # row = (signal_id, symbol, stored_side, qty, entry_px, stop_px, tp_remaining, opened_at)
        _, _, stored_side, _, _, stop_px, _, _ = row

        # allow profile override of atr_mult
        atr_mult = profile.get("atr_mult", self.atr_mult) if profile else self.atr_mult
        new_stop = stop_px

        # tighten on EMA/VWAP reversion
        if stored_side == "BUY":
            if ema_fast_dist < -0.002 or vwap_dist < -0.002:
                new_stop = price - atr_mult * atr
        else:
            if ema_fast_dist > 0.002 or vwap_dist > 0.002:
                new_stop = price + atr_mult * atr

        # widen if ATR spikes >30% vs. entry ATR
        entry_atr = self._entry_atr.get(symbol, atr)
        if atr > 1.3 * entry_atr:
            if stored_side == "BUY":
                new_stop = price - atr_mult * atr
            else:
                new_stop = price + atr_mult * atr

        # persist the change if we moved the stop
        if new_stop != stop_px:
            self.store.update_stop(symbol, new_stop)
            return new_stop

        return None
