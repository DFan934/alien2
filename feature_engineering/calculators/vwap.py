############################
# feature_engineering/calculators/vwap.py
############################
"""Volume‑Weighted Average Price deviation (% from session VWAP)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Calculator, RollingCalculatorMixin, BaseCalculator


class VWAPCalculator(RollingCalculatorMixin, BaseCalculator):
    name = "vwap_delta"
    # Session‑cumulative → no rolling lookback, but we reset at each trading day.
    lookback = 0

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:  # noqa: D401
        if not {"close", "volume"}.issubset(df.columns):
            raise KeyError("VWAPCalculator requires close & volume columns")

        # Assume df is one symbol, chronologically sorted minute bars.
        # Identify session boundaries by date change (UTC)
        day = df["timestamp"].dt.date
        cum_px_vol = (df["close"] * df["volume"]).groupby(day).cumsum()
        cum_vol = df["volume"].groupby(day).cumsum().replace(0, np.nan)
        vwap = cum_px_vol / cum_vol
        delta = (df["close"] - vwap) / vwap
        return pd.DataFrame({self.name: delta.astype("float32")})