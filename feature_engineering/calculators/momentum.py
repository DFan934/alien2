############################
# feature_engineering/calculators/momentum.py
############################
from __future__ import annotations

import pandas as pd

from .base import RollingCalculatorMixin


class MomentumCalculator(RollingCalculatorMixin):
    """N‑period rate of change (%)."""

    def __init__(self, period: int = 20):
        self.period = period
        self.name = f"roc_{period}"
        self.lookback = period

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        roc = df["close"].pct_change(self.period)
        return pd.DataFrame({self.name: roc.astype("float32")})