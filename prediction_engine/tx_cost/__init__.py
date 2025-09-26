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
    #_IMPACT_COEFF: float = 9e-5          # κ
    #_DEFAULT_SPREAD_CENTS: float = 2.0   # ½-spread when no quote supplied
    #_COMMISSION: float = 0.002           # $/share
    #_IMPACT_COEFF: float = 0.0
    #_DEFAULT_SPREAD_CENTS: float = 0.0   # ½-spread when no quote supplied
    #_COMMISSION: float = 0.000
    #_VOL_COEFF = 0.5  # tune later
    #_SPREAD_FALLBACK = 0.003  # 30 bp round-trip for microcaps

    # Calibrated-but-conservative defaults
    # impact is modeled per-share via sqrt(q/ADV) — ~0.5 bps equivalent
    '''_IMPACT_COEFF: float = 5e-5  # κ
    _DEFAULT_SPREAD_CENTS: float = 1.5  # ½-spread fallback in ¢
    _COMMISSION: float = 0.00005  # $/share (e.g., $0.005)
    _SPREAD_FALLBACK = 0.00015  # 1.5¢ half-spread fallback (USD)
    '''

    # --- CHANGED: Constants updated per the patch ---
    # Commission: five mills per share ($0.005/share)
    _COMMISSION = 0.005  # $/share
    # Half-spread defaults:
    _DEFAULT_SPREAD_CENTS = 1.5  # cents
    _SPREAD_FALLBACK = 0.015  # dollars (1.5 cents)

    # Kept from original for the cost() method
    _IMPACT_COEFF: float = 5e-5  # κ
    _VOL_COEFF = 0.5


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
        **kwargs,
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

        if adv_pct is None and "adv_percentile" in kwargs:
            adv_pct = kwargs.pop("adv_percentile")

        # --- spread + commission ------------------------------------- #
        #hs = half_spread if half_spread is not None else self._DEFAULT_SPREAD_CENTS * 0.01
        hs = half_spread if half_spread is not None else self._DEFAULT_SPREAD_CENTS * 0.01

        #commission = self._COMMISSION
        c  = self._COMMISSION

        # --- add volatility‐based slippage term ---------------- #
        # caller may optionally pass `volatility` (std‐dev of returns) in adv_pct kw
        #vol = adv_pct if isinstance(adv_pct, float) and vol_kwarg_provided else 0.0
        #vol_slippage = getattr(self, "_VOL_COEFF", 0.5) * vol

        # Using getattr for robustness, as shown in the patch
        impact = getattr(self, "_IMPACT_COEFF", 0.0) * (adv_pct or 0.0)
        vol_slip = getattr(self, "_VOL_COEFF", 0.0) * (volatility or 0.0) if volatility is not None else 0.0

        per_share = hs + c + impact + vol_slip

        # The original file returned per-share cost, so we keep that behavior
        # instead of multiplying by abs(qty) as the patch's comments imply.
        return float(per_share)

    # ------------------------------------------------------------------ #
    # Convenience alias (back-compat with old code)                      #
    # ------------------------------------------------------------------ #
    '''def estimate(self,
                 *,
                                  half_spread: float | None = None,
                                  adv_percentile: float | None = None) -> float:


    # 1. spread component  (if caller didn’t provide one)
        hs = half_spread if half_spread is not None else self._SPREAD_FALLBACK

    # 2. slippage: 0 bp at ADV≤5 %, ramps to 15 bp at ADV=20 %

        if adv_percentile is None:
            slip = 0.0000
        else:
            slip = 0.00015 * max(0.0, min((adv_percentile - 5) / 15, 1.0))


        #return hs + slip
        return float(hs + self._COMMISSION + slip)'''

    def estimate(
            self,
            *,
            half_spread: float | None = None,
            adv_pct: float | None = None,  # CHANGED: Parameter renamed from adv_percentile
            **kwargs,
    ) -> float:
        """
        Return an *estimate* of per-share cost to use in EV or sizing logic.
        """

        # Back-compat: allow callers to pass adv_percentile=...
        if adv_pct is None and "adv_percentile" in kwargs:
            adv_pct = kwargs.pop("adv_percentile")

        # --- CHANGED: Logic replaced with the patch's simplified model ---
        hs = half_spread if half_spread is not None else self._SPREAD_FALLBACK
        c = self._COMMISSION
        impact = getattr(self, "_IMPACT_COEFF", 0.0) * (adv_pct or 0.0)

        return float(hs + c + impact)

# --- NEW: Explicit zero-cost model for debug / unit tests --------------
class NoCostModel(BaseCostModel):
    """Always returns zero per-share cost (useful for debug runs)."""
    def cost(self, qty: float = 1.0, *, half_spread: float | None = None,
             adv_pct: float | None = None, **kwargs) -> float:
        return 0.0
    def estimate(self, *, half_spread: float | None = None,
                 adv_percentile: float | None = None, **kwargs) -> float:
        return 0.0

# ---------------------------------------------------------------------------#
# Legacy free-function shim  – forwards to singleton instance                #
# ---------------------------------------------------------------------------#
_basic = BasicCostModel()  # stateless singleton


def estimate(*args, **kwargs):  # noqa: D401
    """Compatibility wrapper for old `tx_cost.estimate` import paths."""
    return _basic.estimate(*args, **kwargs)
