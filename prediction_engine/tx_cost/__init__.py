"""
Transaction-cost models
=======================

`BaseCostModel` – abstract interface
`BasicCostModel` – square-root-impact Slippage model (old `_tx_cost.basic`)
The module still exposes the free-function **estimate(...)** so legacy code
continues to work (`EVEngine` expects it).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

__all__ = ("BaseCostModel", "BasicCostModel", "estimate")


class BaseCostModel(ABC):
    """Every cost-model must implement these two helpers."""

    @abstractmethod
    def cost(self, qty: float, adv_pct: float | None = None) -> float: ...
    @abstractmethod
    def estimate(self, adv_percentile: float | None = None) -> float: ...


class BasicCostModel(BaseCostModel):
    """Square-root market-impact model, calibrated nightly."""

    _IMPACT_COEFF: float = 9e-5  # <- tuned by nightly calibration

    def cost(self, qty: float = 1.0, adv_pct: float | None = None) -> float:
        adv = 1.0  # unit ADV normalisation – fine for per-share usage
        frac = np.clip(qty / adv, 1e-6, 1.0)
        return float(self._IMPACT_COEFF * np.sqrt(frac))

    def estimate(self, adv_percentile: float | None = None) -> float:
        # delegate to cost() with 1-share notional
        return self.cost(1.0, adv_percentile)


# ------------------------------------------------------------------#
# Back-compat helpers – keep the free function used in older code   #
# ------------------------------------------------------------------#
_basic = BasicCostModel()          # singleton – stateless

def estimate(*args, **kwargs):     # noqa: D401  (simple alias)
    """Compatibility shim – forwards to `BasicCostModel.estimate()`."""
    return _basic.estimate(*args, **kwargs)
