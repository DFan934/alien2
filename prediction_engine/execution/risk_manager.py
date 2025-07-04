# ---------------------------------------------------------------------------
# execution/risk_manager.py  –  zero‑division‑safe, ATR aware
# ---------------------------------------------------------------------------
"""Risk management utilities – updated to avoid divide‑by‑zero when price or ATR
is missing / zero."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RiskManager:
    account_equity: float          # total account value in dollars
    max_leverage: float = 2.0      # notional / equity cap
    risk_per_trade: float = 0.001  # fraction of equity risked per trade (0.1 %)
    atr_multiplier: float = 1.5    # used for stop distance

    _symbol_atr: Dict[str, float] = field(default_factory=dict, init=False)
    _peak_equity: float = field(init=False)

    def __post_init__(self):
        self._peak_equity = self.account_equity

    # --------------------------------------------------------------------
    # Realtime updates
    # --------------------------------------------------------------------
    def update_atr(self, symbol: str, atr: float):
        if atr is not None and atr > 0:
            self._symbol_atr[symbol] = atr

    # --------------------------------------------------------------------
    # Position sizing helpers
    # --------------------------------------------------------------------
    def _dollar_risk(self) -> float:
        return self.account_equity * self.risk_per_trade

    def desired_size(self, symbol: str, price: float) -> int:
        """Return share qty; guards against zero price / ATR."""
        if price is None or price <= 0 or math.isnan(price):
            return 0  # skip if price bad

        atr = self._symbol_atr.get(symbol)
        if atr is None or atr <= 0 or math.isnan(atr):
            stop_dist = max(0.01 * price, 1e-6)  # fallback 1 % or 1e‑6
        else:
            stop_dist = max(self.atr_multiplier * atr, 1e-6)

        qty = math.floor(self._dollar_risk() / stop_dist)
        max_qty = math.floor(self.account_equity * self.max_leverage / price)
        return max(1, min(qty, max_qty))

    # --------------------------------------------------------------------
    # Drawdown tracking
    # --------------------------------------------------------------------
    def on_closed_trade(self, pnl: float):
        self.account_equity += pnl
        self._peak_equity = max(self._peak_equity, self.account_equity)

    def drawdown(self) -> float:
        return 1.0 - self.account_equity / self._peak_equity
