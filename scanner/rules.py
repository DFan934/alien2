# ---------------------------------------------------------------------------
# FILE: scanner/rules.py
# ---------------------------------------------------------------------------
"""Stateless helper functions that return pandas.Series[bool] masks.
All rules assume *df* has a DatetimeIndex and the following columns:
    open, close, high, low, volume, vwap, rvol (20‑day)
Additional indicator columns may be present but are ignored.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

__all__ = [
    "gap_percent",
    "gap_up",
    "gap_down",
    "high_rvol",
    "bullish_premarket_momentum",
]


def gap_percent(df: pd.DataFrame) -> pd.Series:
    """Return (open − prev_close)/prev_close for each bar."""
    prev_close = df["close"].shift(1)
    return (df["open"] - prev_close) / prev_close


def gap_up(df: pd.DataFrame, *, pct: float = 0.02) -> pd.Series:
    """Boolean mask – bars gapping **up** ≥ *pct* (default 2 %)."""
    gp = gap_percent(df)
    return gp >= pct


def gap_down(df: pd.DataFrame, *, pct: float = 0.02) -> pd.Series:  # noqa: D401
    """Mask – bars gapping **down** ≥ *pct*."""
    gp = gap_percent(df)
    return gp <= -pct


def high_rvol(df: pd.DataFrame, *, thresh: float = 2.0) -> pd.Series:
    """Mask – *relative volume* column exceeds *thresh* (≥2 by default)."""
    if "rvol" not in df:
        raise KeyError("DataFrame missing 'rvol' column required for high_rvol rule")
    return df["rvol"] >= thresh


def bullish_premarket_momentum(df: pd.DataFrame, *, window: int = 3) -> pd.Series:
    """Simple proxy: last *window* bars close > open ⇒ bullish momentum.
    Premarket flag is left to the caller (feed should already be filtered
    to pre‑market bars if desired).
    """
    closes = df["close"].rolling(window).apply(lambda x: np.all(np.diff(x) > 0), raw=False)
    return closes.fillna(False).astype(bool)
