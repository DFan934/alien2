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
        Trailing / widening stop logic with hard gap-fail check.
        """
        row = self.store.get_open_symbol(symbol)
        if row is None:
            return None

        # unpack
        _, _, stored_side, _, _, stop_px, _, _ = row

        # ── 0. Hard gap-fail: price has already crossed the stop ─────────
        # Force an immediate update so the exit manager can flatten.
        if (stored_side == "BUY" and price < stop_px) or \
                (stored_side == "SELL" and price > stop_px):
            self.store.update_stop(symbol, price)
            return price  # we’re done – no further adjustments

        # ── 1. Normal trailing logic ────────────────────────────────────
        atr_mult = profile.get("atr_mult", self.atr_mult) if profile else self.atr_mult
        new_stop = stop_px

        # tighten on EMA / VWAP mean-reversion
        if stored_side == "BUY":
            if ema_fast_dist < -0.002 or vwap_dist < -0.002:
                new_stop = price - atr_mult * atr
        else:
            if ema_fast_dist > 0.002 or vwap_dist > 0.002:
                new_stop = price + atr_mult * atr

        # widen if ATR has expanded > 30 %
        entry_atr = self._entry_atr.get(symbol, atr)
        if atr > 1.3 * entry_atr:
            if stored_side == "BUY":
                new_stop = price - atr_mult * atr
            else:
                new_stop = price + atr_mult * atr

        #if new_stop != stop_px:
        #    self.store.update_stop(symbol, new_stop)
        #    return new_stop

        # ── HARD RISK CAP ── keep stop within ±2 % of entry price
        risk_cap = 0.02 * price
        if stored_side == "BUY":
            new_stop = max(new_stop, price - risk_cap)
        else:
            new_stop = min(new_stop, price + risk_cap)
        if new_stop != stop_px:
            self.store.update_stop(symbol, new_stop)
            return new_stop

        return None
