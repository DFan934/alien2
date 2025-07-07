# --- feature_engineering/calculators/candlestick_score.py ---
"""Candlestick trend‑versus‑indecision score (Level‑1 pseudo order‑flow).

* body length → momentum
* wick length → indecision / volatility

The rolling trend_score = mean(momentum) / (mean(indecision)+ε).
The module exposes **both** a functional helper *candlestick_trend_vs_indecision* and a
wrapper class *TrendVsIndecisionCalculator* implementing the same interface as the
other calculator classes in this package so it can live inside ``ALL_CALCULATORS``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "candlestick_trend_vs_indecision",
    "TrendVsIndecisionCalculator",
]

def candlestick_trend_vs_indecision(
    df: pd.DataFrame,
    window: int = 4,
    eps: float = 1e-6,
) -> pd.Series:
    """Return rolling trend‑vs‑indecision score as a ``pd.Series``.

    Parameters
    ----------
    df : DataFrame with columns [open, high, low, close]
    window : look‑back window length
    eps : numerical stabiliser to avoid division by zero
    """
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")

    body = (df["close"] - df["open"]).abs()
    full_rng = df["high"] - df["low"]
    wick = full_rng - body

    momentum = body / df["close"].shift(1)
    indecision = wick / full_rng.replace(0, np.nan)

    trend = momentum.rolling(window).mean()
    indec = indecision.rolling(window).mean()

    score = trend / (indec + eps)
    score.name = "trend_vs_indecision"
    return score

class TrendVsIndecisionCalculator:
    """Class wrapper so the feature pipeline can treat this like all others."""

    name = "trend_vs_indecision"

    def __call__(self, df: pd.DataFrame) -> pd.Series:  # noqa: D401
        """Return the score Series for integration into feature set."""
        return candlestick_trend_vs_indecision(df)
