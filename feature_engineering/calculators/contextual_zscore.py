# ──────────────────────────────────────────────
# feature_engineering/calculators/contextual_zscore.py
# ──────────────────────────────────────────────
"""Contextual Z‑Score normalisation for body & wick."""
from __future__ import annotations

import pandas as pd
import numpy as np

class ContextualZScoreCalculator:
    name = "contextual_zscore"

    def __init__(self, long_window: int = 250):
        self.long_window = long_window

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        body = (df["close"] - df["open"]).abs()
        full = df["high"] - df["low"]
        wick = full - body

        zb = (body - body.rolling(self.long_window).mean()) / body.rolling(self.long_window).std()
        zw = (wick - wick.rolling(self.long_window).mean()) / wick.rolling(self.long_window).std()
        res = pd.DataFrame({
            "z_body": zb,
            "z_wick": zw,
        }, index=df.index)
        return res