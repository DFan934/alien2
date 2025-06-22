############################
# feature_engineering/calculators/ema.py
############################
"""Exponential moving averages & distance to EMA."""
from __future__ import annotations

import pandas as pd

from .base import Calculator, RollingCalculatorMixin


class EMACalculator(RollingCalculatorMixin):
    def __init__(self, period: int):
        self.period = period
        self.name = f"ema_{period}_dist"
        self.lookback = period * 3  # safe pad for ewm

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if "close" not in df.columns:
            raise KeyError("EMACalculator requires close column")
        ema = df["close"].ewm(span=self.period, adjust=False).mean()
        dist = (df["close"] - ema) / ema
        return pd.DataFrame({self.name: dist.astype("float32")})

class EMA9Calculator(EMACalculator):
    def __init__(self) -> None:
        super().__init__(period=9)

class EMA20Calculator(EMACalculator):
    def __init__(self) -> None:
        super().__init__(period=20)
