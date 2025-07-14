# -----------------------------------------------------------------------------
# execution/risk_manager.py
# -----------------------------------------------------------------------------
"""ATR‑aware Kelly risk sizing + drawdown tracking (M1)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class RiskManager:
    account_equity: float
    max_leverage: float = 2.0
    risk_per_trade: float = 0.001  # 0.1 %
    atr_multiplier: float = 1.5
    safety_fsm: "Optional[object]" = None  # lazy import

    _symbol_atr: Dict[str, float] = field(default_factory=dict, init=False)
    _peak_equity: float = field(init=False)

    # --------------------- lifecycle ----------------------------------------
    def __post_init__(self):
        self._peak_equity = self.account_equity

    # --------------------- real‑time updates ---------------------------------
    def update_atr(self, symbol: str, atr: float):
        if atr and atr > 0:
            self._symbol_atr[symbol] = atr

    # --------------------- sizing helpers ------------------------------------
    def _dollar_risk(self) -> float:
        return self.account_equity * self.risk_per_trade

    def desired_size(self, symbol: str, price: float) -> int:
        if price <= 0 or math.isnan(price):
            return 0
        atr = self._symbol_atr.get(symbol)
        stop_dist = max(self.atr_multiplier * atr, 0.01 * price) if atr else 0.01 * price
        qty = math.floor(self._dollar_risk() / max(stop_dist, 1e-6))
        max_qty = math.floor(self.account_equity * self.max_leverage / price)
        return max(0, min(qty, max_qty))

    def kelly_position(self, mu: float, variance_down: float, price: float, kelly_cap: float = 1.0) -> int:
        if price <= 0 or variance_down <= 0 or mu <= 0:
            return 0
        kelly_f = min(mu / (2 * variance_down), kelly_cap)
        dollar_notional = kelly_f * self.account_equity
        max_notional = self.account_equity * self.max_leverage
        qty = math.floor(min(dollar_notional, max_notional) / price)
        return max(qty, 0)

    # --------------------- drawdown tracking ---------------------------------
    def on_closed_trade(self, pnl: float):
        self.account_equity += pnl
        self._peak_equity = max(self._peak_equity, self.account_equity)
        if self.safety_fsm is not None:
            self.safety_fsm.register_trade(pnl)

    def drawdown(self) -> float:
        return 1.0 - self.account_equity / self._peak_equity

    @staticmethod
    def scale_variance(var: float, adv_percentile: float | None) -> float:
        if adv_percentile is None:
            return var
        mult = 1.0 + 0.5 * min(max((adv_percentile - 5) / 15, 0.0), 1.0)
        return var * mult



    # --------------------------------------------------------------------
    # Value-at-risk helper (for SafetyFSM & metrics)
    # --------------------------------------------------------------------
    def position_value_at_risk(self, entry_px: float, stop_px: float, qty: int) -> float:
        """Absolute dollar risk of an open position."""
        return abs(entry_px - stop_px) * qty



