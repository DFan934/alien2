# ---------------------------------------------------------------------------
# __init__.py – export classes & logger
# ---------------------------------------------------------------------------

"""Expose every calculator and ALL_CALCULATORS for orchestration."""
from .calculators.vwap import VWAPCalculator
from .calculators.rvol import RVOLCalculator
from .calculators.ema import EMA9Calculator, EMA20Calculator
from .calculators.momentum import MomentumCalculator
from .calculators.atr import ATRCalculator
from .calculators.adx import ADXCalculator

ALL_CALCULATORS = [
    VWAPCalculator(),
    RVOLCalculator(),
    EMA9Calculator(),
    EMA20Calculator(),
    MomentumCalculator(),
    ATRCalculator(),
    ADXCalculator(),
]

__all__ = [
    "VWAPCalculator",
    "RVOLCalculator",
    "EMA9Calculator",
    "EMA20Calculator",
    "MomentumCalculator",
    "ATRCalculator",
    "ADXCalculator",
    "ALL_CALCULATORS",
]