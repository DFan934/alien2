# ---------------------------
# FILE: prediction_engine/execution/__init__.py
# ---------------------------
"""Execution sub‑package."""
from __future__ import annotations

__all__ = ["manager", "risk_manager"]


# ---------------------------
# FILE: prediction_engine/execution/risk_manager.py
# ---------------------------
"""Simple risk sizing – placeholder."""

import math


class RiskManager:
    _EQUITY = 100_000
    _MAX_RISK = 0.02

    def desired_size(self, symbol: str) -> int:
        return int(self._EQUITY * self._MAX_RISK)

    def position_size(self, symbol: str, ev_mean: float, ev_var: float) -> int:
        k = max(min(ev_mean / (ev_var + 1e-6), 1.0), 0.0)
        return int(self.desired_size(symbol) * k)
