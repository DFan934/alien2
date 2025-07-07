############################
# feature_engineering/calculators/atr.py
############################
from __future__ import annotations

import pandas as pd

from .base import RollingCalculatorMixin, BaseCalculator


class ATRCalculator(RollingCalculatorMixin, BaseCalculator):
    """Average True Range (volatility proxy)."""

    def __init__(self, period: int = 14):
        self.period = period
        self.name = f"atr_{period}"
        self.lookback = period

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(self.period, min_periods=1).mean()
        return pd.DataFrame({self.name: atr.astype("float32")})