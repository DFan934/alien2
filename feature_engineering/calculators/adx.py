############################
# feature_engineering/calculators/adx.py
############################
from __future__ import annotations

import pandas as pd

from .base import RollingCalculatorMixin


class ADXCalculator(RollingCalculatorMixin):
    """Average Directional Index (trend strength)."""

    def __init__(self, period: int = 14):
        self.period = period
        self.name = f"adx_{period}"
        self.lookback = period * 2

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        up_move = df["high"].diff()
        down_move = -df["low"].diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        tr = (
            pd.concat([(df["high"] - df["low"]), (df["high"] - df["close"].shift()).abs(), (df["low"] - df["close"].shift()).abs()], axis=1)
            .max(axis=1)
            .rolling(self.period)
            .sum()
        )
        plus_di = 100 * plus_dm.rolling(self.period).sum() / tr.replace(0, pd.NA)
        minus_di = 100 * minus_dm.rolling(self.period).sum() / tr.replace(0, pd.NA)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, pd.NA)) * 100
        adx = dx.rolling(self.period).mean()
        return pd.DataFrame({self.name: adx.astype("float32")})
