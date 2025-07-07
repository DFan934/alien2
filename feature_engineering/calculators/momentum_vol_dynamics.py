# ──────────────────────────────────────────────
# feature_engineering/calculators/momentum_vol_dynamics.py
# ──────────────────────────────────────────────
"""Momentum‑vs‑Volatility Dynamics Calculator."""
from __future__ import annotations

import pandas as pd
import numpy as np
from scipy.stats import linregress

class MomentumVolatilityDynamicsCalculator:
    name = "momvol_dynamics"

    def __init__(self, window: int = 4):
        self.window = window

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        req = {"open", "high", "low", "close"}
        if not req.issubset(df.columns):
            raise ValueError("Missing OHLC columns")

        body = (df["close"] - df["open"]).abs()
        full = df["high"] - df["low"]
        wick = full - body
        vr = body / wick.replace(0, np.nan)

        out = pd.DataFrame(index=df.index)
        out["momentum_slope"] = (
            body.rolling(self.window)
            .apply(lambda x: linregress(range(len(x)), x).slope, raw=True)
        )
        out["wick_slope"] = (
            wick.rolling(self.window)
            .apply(lambda x: linregress(range(len(x)), x).slope, raw=True)
        )
        out["ratio_slope"] = (
            vr.rolling(self.window)
            .apply(lambda x: linregress(range(len(x)), x).slope, raw=True)
        )
        out["pattern_entropy"] = (
            np.sign(df["close"].diff())
            .rolling(self.window)
            .apply(lambda x: -np.nansum(np.bincount((x > 0).astype(int), minlength=2) / len(x)
                                         * np.log2(np.bincount((x > 0).astype(int), minlength=2) / len(x) + 1e-9)),
                   raw=False)
        )
        return out