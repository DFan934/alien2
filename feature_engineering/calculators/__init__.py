# ──────────────────────────────────────────────
# feature_engineering/calculators/__init__.py (PATCHED)
# ──────────────────────────────────────────────
"""Expose calculator classes & maintain ``ALL_CALCULATORS``."""
from .vwap import VWAPCalculator
from .rvol import RVOLCalculator
from .ema import EMA9Calculator, EMA20Calculator
from .momentum import MomentumCalculator
from .atr import ATRCalculator
from .adx import ADXCalculator
from .candlestick_score import TrendVsIndecisionCalculator
from .multi_tf_trend import MultiTFTrendCalculator
from .momentum_vol_dynamics import MomentumVolatilityDynamicsCalculator
from .contextual_zscore import ContextualZScoreCalculator

ALL_CALCULATORS = [
    VWAPCalculator(),
    RVOLCalculator(),
    EMA9Calculator(),
    EMA20Calculator(),
    MomentumCalculator(),
    ATRCalculator(),
    ADXCalculator(),
    TrendVsIndecisionCalculator(),
    MultiTFTrendCalculator(intervals=[2, 4, 8]),
    MomentumVolatilityDynamicsCalculator(window=4),
    ContextualZScoreCalculator(long_window=250),
]

__all__ = [cls.__class__.__name__ for cls in ALL_CALCULATORS] + ["ALL_CALCULATORS"]