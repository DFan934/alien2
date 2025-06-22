##############################################
# feature_engineering/calculators/__init__.py  (NEW)
##############################################
"""Expose every calculator class and a convenience list ``ALL_CALCULATORS``.
Add new calculators here so pipelines pick them up automatically."""
from .vwap import VWAPCalculator
from .rvol import RVOLCalculator
from .ema import EMA9Calculator, EMA20Calculator
from .momentum import MomentumCalculator
from .atr import ATRCalculator
from .adx import ADXCalculator

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
