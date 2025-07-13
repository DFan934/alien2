# ---------------------------------------------------------------------------
# FILE: scanner/detectors.py
# ---------------------------------------------------------------------------
"""Object‑oriented wrappers around *rules* so that thresholds are bound
once and the detector becomes a callable **mask(df) → pd.Series[bool]**.
"""
from __future__ import annotations

"""Setup‑detector helpers (UPDATED).

Only the *SectorMomentumDetector* section changed – rest of file is unchanged.
"""

from dataclasses import dataclass, field
from typing import Callable
import pandas as pd

from .sector_feed import CACHE  # <-- new dependency

__all__ = [
    "GapDetector",
    "HighRVOLDetector",
    "BullishPremarketMomentumDetector",
    "SectorMomentumDetector",
    "BullishEngulfingDetector",
    "CompositeDetector",
]

# Existing simple detectors ---------------------------------------------------

def _pct_gap(df: pd.DataFrame) -> pd.Series:
    return (df["open"] - df["prev_close"]) / df["prev_close"]

def _rvol(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    return df["volume"] / df["volume"].rolling(lookback).mean()

@dataclass(slots=True)
class GapDetector:
    pct: float = 0.02
    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return _pct_gap(df).abs() >= self.pct

@dataclass(slots=True)
class HighRVOLDetector:
    thresh: float = 2.0
    lookback: int = 20
    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return _rvol(df, self.lookback) >= self.thresh

@dataclass(slots=True)
class BullishPremarketMomentumDetector:
    window: int = 15  # minutes – simplistic
    def __call__(self, df: pd.DataFrame) -> pd.Series:
        pct = (df["close"] / df["close"].shift(self.window) - 1.0).fillna(0)
        return pct > 0.01

# NEW / REVISED detectors -----------------------------------------------------

@dataclass(slots=True)
class SectorMomentumDetector:
    """Passes when stock direction aligns with its sector ETF trend."""
    sector: str  # e.g. "XLE" – energy ETF symbol
    window: int = 5  # minutes for ETF trend
    min_corr: float = 0.3  # simplistic similarity measure

    async def _sector_trend(self) -> float:
        px_now = await CACHE.get(self.sector)
        # Fallback to NaN means detector returns all‑False
        if px_now is None or px_now != px_now:  # NaN check
            return float("nan")
        # Pull a synthetic *previous* price window minutes ago (stub)
        # Real impl would buffer prices; stub: −0.05 % drift
        return (px_now * 0.9995)  # pretend small up‑trend

    async def __call__(self, df: pd.DataFrame) -> pd.Series:  # type: ignore[override]
        trend_px0 = await self._sector_trend()
        if trend_px0 != trend_px0:
            return pd.Series(False, index=df.index)
        trend = (await CACHE.get(self.sector) - trend_px0) / trend_px0
        # Stock’s own return over same window
        stock_ret = (df["close"] / df["close"].shift(self.window) - 1.0).fillna(0)
        # Simple sign alignment check
        aligned = (trend >= 0) & (stock_ret >= 0) | (trend < 0) & (stock_ret < 0)
        return aligned & (stock_ret.abs() > 0.002)


@dataclass(slots=True)
class BullishEngulfingDetector:
    def __call__(self, df: pd.DataFrame) -> pd.Series:
        o, c = df["open"], df["close"]
        prev_o, prev_c = o.shift(1), c.shift(1)
        return (prev_c < prev_o) & (c > o) & (c > prev_o) & (o < prev_c)


@dataclass(slots=True)
class CompositeDetector:
    """Logical‑AND reducer over sub‑detectors (sync **or** async)."""
    sub: list[Callable[[pd.DataFrame], "pd.Series | Awaitable[pd.Series]"]] = field(default_factory=list)

    async def __call__(self, df: pd.DataFrame) -> pd.Series:  # noqa: C901
        import inspect, asyncio
        mask = pd.Series(True, index=df.index)
        for det in self.sub:
            res = det(df)  # may be coroutine
            if inspect.iscoroutine(res):
                res = await res  # type: ignore[assignment]
            mask &= res
            if not mask.any():
                break  # early‑exit
        return mask
