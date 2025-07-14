# -----------------------------------------------------------------------------
# execution/order_builder.py
# -----------------------------------------------------------------------------
"""Builds OrderEvent objects with ATR‑based stops & TP tiers (M1)."""
from __future__ import annotations

from math import floor
from typing import Tuple

from core.contracts import OrderEvent, TradeSignal
from execution.risk_manager import RiskManager


class OrderBuilder:
    ATR_MULTIPLIER = 1.5
    TP_TIERS = (0.02, 0.04)  # +2 % / +4 %

    def __init__(self, risk_manager: RiskManager):
        self.rm = risk_manager

    def build_entry(self, sig: TradeSignal) -> Tuple[OrderEvent, float]:
        stop_dist = max(sig.atr * self.ATR_MULTIPLIER, 0.01 * sig.price)
        stop_px = sig.price - stop_dist if sig.side == "BUY" else sig.price + stop_dist
        qty = self.rm.desired_size(sig.symbol, sig.price)
        tp_tiers = [sig.price * (1 + t if sig.side == "BUY" else 1 - t) for t in self.TP_TIERS]
        order = OrderEvent(
            signal_id=sig.id,
            symbol=sig.symbol,
            side=sig.side,
            qty=qty,
            stop_px=stop_px,
            tp_tiers=tp_tiers,
        )
        return order, stop_px