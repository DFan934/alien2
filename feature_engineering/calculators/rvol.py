############################
# feature_engineering/calculators/rvol.py
############################
"""Relative volume: current volume vs N‑day mean for same minute index."""
from __future__ import annotations

import pandas as pd

from .base import Calculator, RollingCalculatorMixin


class RVOLCalculator(RollingCalculatorMixin):
    def __init__(self, lookback_days: int = 20):
        self.name = f"rvol_{lookback_days}d"
        # 390 min per session (US equities) → store days for window calc
        self.lookback = lookback_days * 390
        self._days = lookback_days

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if "volume" not in df.columns:
            raise KeyError("RVOLCalculator requires volume column")

        # Minute index within session (0‑389)
        idx_in_day = (df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute) - 570  # 9:30 open
        avg_vol = (
            df["volume"].groupby(idx_in_day).transform(lambda x: x.rolling(self._days, min_periods=1).mean())
        )
        rvol = df["volume"] / avg_vol.replace(0, pd.NA)
        return pd.DataFrame({self.name: rvol.astype("float32")})
