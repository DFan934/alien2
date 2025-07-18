# =============================================================================
# FILE: tx_cost/__init__.py
# =============================================================================
"""
Transaction-cost models
=======================

`BaseCostModel` – abstract interface
`BasicCostModel` – half-spread + commission + square-root-impact model

The module still exports the free function **estimate(…)** so legacy code
(`EVEngine`, test-benches, etc.) keeps working unchanged.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = ("BaseCostModel", "BasicCostModel", "estimate")


# ---------------------------------------------------------------------------#
# Abstract interface                                                         #
# ---------------------------------------------------------------------------#
class BaseCostModel(ABC):
    """All cost-models expose the same two helpers."""

    @abstractmethod
    def cost(
        self,
        qty: float = 1.0,
        *,
        half_spread: float | None = None,
        adv_pct: float | None = None,
    ) -> float: ...

    @abstractmethod
    def estimate(
        self,
        *,
        half_spread: float | None = None,
        adv_percentile: float | None = None,
    ) -> float: ...


# ---------------------------------------------------------------------------#
# Default implementation – calibrated nightly                                #
# ---------------------------------------------------------------------------#
class BasicCostModel(BaseCostModel):
    """
    Realistic per-share transaction cost:

        cost = half_spread  +  commission  +  impact

    * **half_spread** – default 2¢; caller can pass live quote for accuracy.
    * **commission**   – broker fee, default 0.2 ¢/share (≈ 0.002 USD).
    * **impact**       – square-root model :math:`κ·√(q/ADV)`.
    """

    # Nightly-calibrated constants (override before back-test if needed)
    _IMPACT_COEFF: float = 9e-5          # κ
    _DEFAULT_SPREAD_CENTS: float = 2.0   # ½-spread when no quote supplied
    _COMMISSION: float = 0.002           # $/share
    _VOL_COEFF = 0.5  # tune later

    # ------------------------------------------------------------------ #
    # Core                                                               #
    # ------------------------------------------------------------------ #
    def cost(
        self,
        qty: float = 1.0,
        *,
        half_spread: float | None = None,
        adv_pct: float | None = None,
        volatility: float | None = None,    # NEW: σ of 1‑min returns
    ) -> float:
        """
        Parameters
        ----------
        qty
            Order size in shares.
        half_spread
            Live half-spread in **USD** (not ¢).  Falls back to 2 ¢ if omitted.
        adv_pct
            Order size **as % of ADV** (0–100).  When supplied, impact is scaled
            by that liquidity tier; otherwise unit ADV is assumed.
        """
        # --- spread + commission ------------------------------------- #
        hs = half_spread if half_spread is not None else self._DEFAULT_SPREAD_CENTS * 0.01
        commission = self._COMMISSION

        # --- add volatility‐based slippage term ---------------- #
        # caller may optionally pass `volatility` (std‐dev of returns) in adv_pct kw
        #vol = adv_pct if isinstance(adv_pct, float) and vol_kwarg_provided else 0.0
        #vol_slippage = getattr(self, "_VOL_COEFF", 0.5) * vol

        # --- market-impact (square-root) ----------------------------- #
        if adv_pct is None:
            frac = 1.0  # assume 1 × ADV normalisation (unit share cost)
        else:
            # adv_pct is “% of ADV” for *this* order; clamp to [1e-6, 100]
            frac = np.clip(adv_pct / 100.0, 1e-6, 1.0)

        impact = self._IMPACT_COEFF * np.sqrt(frac)

        # --- volatility‑scaled queue slippage --------------------- #
        vol_slippage = 0.0

        if volatility is not None and np.isfinite(volatility):
                        vol_slippage = self._VOL_COEFF * volatility


        return float(hs + commission + impact + vol_slippage)

    # ------------------------------------------------------------------ #
    # Convenience alias (back-compat with old code)                      #
    # ------------------------------------------------------------------ #
    def estimate(
        self,
        *,
        half_spread: float | None = None,
        adv_percentile: float | None = None,
    ) -> float:
        """Delegates to :meth:`cost` using a unit share."""
        return self.cost(
            1.0,
            half_spread=half_spread,
            adv_pct=adv_percentile,
            volatility=None,
        )


# ---------------------------------------------------------------------------#
# Legacy free-function shim  – forwards to singleton instance                #
# ---------------------------------------------------------------------------#
_basic = BasicCostModel()  # stateless singleton


def estimate(*args, **kwargs):  # noqa: D401
    """Compatibility wrapper for old `tx_cost.estimate` import paths."""
    return _basic.estimate(*args, **kwargs)
