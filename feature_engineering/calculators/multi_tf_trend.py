# ──────────────────────────────────────────────
# feature_engineering/calculators/multi_tf_trend.py
# ──────────────────────────────────────────────
"""Multi‑time‑frame Trend‑vs‑Indecision plus Fake‑out count.

Given a base 1‑minute OHLC DataFrame, this calculator:
1. Resamples to user‑supplied intervals (e.g. 2, 4, 8 minutes).
2. Reuses *candlestick_trend_vs_indecision* on each frame.
3. Computes **fakeout_count** – the number of score sign flips on the 1‑min
   series before the coarse series flips.

Outputs:
    • trend_vs_indecision_<N>m        (for each interval)
    • fakeout_count_<N>m
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Iterable, List

from .candlestick_score import candlestick_trend_vs_indecision

class MultiTFTrendCalculator:
    name = "multi_tf_trend"

    def __init__(self, intervals: Iterable[int] = (2, 4, 8), base_freq: str = "1T"):
        self.intervals: List[int] = list(intervals)
        self.base_freq = base_freq  # assumed 1‑minute bars

    # ──────────────────────────────────────────
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if not (df.index.freqstr or df.index.inferred_freq):
            raise ValueError("DataFrame index must be a DatetimeIndex with freq.")

        # Base score on 1‑minute bars
        base_score = candlestick_trend_vs_indecision(df)
        features = pd.DataFrame(index=df.index)
        features["trend_vs_indecision_1m"] = base_score

        # Iterate over coarser frames
        for n in self.intervals:
            rule = f"{n}T"
            resampled = (
                df[["open", "high", "low", "close"]]
                .resample(rule, label="right", closed="right")
                .agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                })
                .dropna()
            )
            score = candlestick_trend_vs_indecision(resampled)
            score = score.reindex(df.index).ffill()
            col = f"trend_vs_indecision_{n}m"
            features[col] = score

            # Fake‑out count: flips in 1‑m before coarse flip
            flips = base_score.apply(np.sign).diff().ne(0).astype(int)
            coarse_flip = score.apply(np.sign).diff().ne(0).astype(int)
            window = n  # minutes
            fakeout = (
                flips.rolling(window).sum() * coarse_flip
            )  # count only at coarse flip
            features[f"fakeout_count_{n}m"] = fakeout.fillna(0)

        return features
