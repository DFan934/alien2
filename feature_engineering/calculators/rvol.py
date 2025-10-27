############################
# feature_engineering/calculators/rvol.py
############################
"""Relative volume: current volume vs N‑day mean for same minute index."""
from __future__ import annotations

import pandas as pd

from .base import Calculator, RollingCalculatorMixin, BaseCalculator
from ..utils.calendar import session_id, minutes_since_open


class RVOLCalculator(RollingCalculatorMixin, BaseCalculator):
    def __init__(self, lookback_days: int = 20):
        self.name = f"rvol_{lookback_days}d"
        # 390 min per session (US equities) → store days for window calc
        self.lookback = lookback_days * 390
        self._days = lookback_days

    '''def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if "volume" not in df.columns:
            raise KeyError("RVOLCalculator requires volume column")

        # Minute index within session (0‑389)
        idx_in_day = (df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute) - 570  # 9:30 open
        avg_vol = (
            df["volume"].groupby(idx_in_day).transform(lambda x: x.rolling(self._days, min_periods=1).mean())
        )
        rvol = df["volume"] / avg_vol.replace(0, pd.NA)
        return pd.DataFrame({self.name: rvol.astype("float32")})
    '''

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not {"timestamp", "symbol", "volume"}.issubset(df.columns):
            raise KeyError("RVOLCalculator requires timestamp, symbol, volume")

        # Get cadence from settings (preferred) or infer from this chunk
        try:
            from feature_engineering.config import settings
            bar_seconds = int(getattr(settings, "bar_seconds", 60))
        except Exception:
            bar_seconds = 60

        # If not set, infer from data as a fallback
        if not bar_seconds:
            diffs = df.groupby("symbol")["timestamp"].diff().dropna().dt.total_seconds()
            bar_seconds = int(round(float(diffs.median()))) if len(diffs) else 60

        # Slot index within the RTH session (0..N-1), cadence-aware
        from ..utils.calendar import session_id, slots_since_open
        slot = slots_since_open(df["timestamp"], bar_seconds=bar_seconds)  # int32
        sess = session_id(df["timestamp"])

        # Keep only RTH slots (slot >= 0)
        base = df.loc[slot >= 0, ["symbol", "volume"]].copy()
        base["slot"] = slot[slot >= 0].values
        base["sess"] = sess[slot >= 0].values

        # Sort to ensure rolling by session order is stable
        base = base.sort_values(["symbol", "slot", "sess"])

        # Rolling mean volume across the last N sessions per (symbol, slot)
        baseline = (
            base.groupby(["symbol", "slot"])["volume"]
            .transform(lambda x: x.rolling(self._days, min_periods=1).mean())
            .astype("float32")
        )
        base["baseline"] = baseline.replace(0.0, pd.NA)

        # Join back and compute RVOL = vol / baseline
        out = pd.DataFrame(index=df.index)
        out[self.name] = pd.NA
        mask = (slot >= 0)
        out.loc[mask, self.name] = (
                df.loc[mask, "volume"].astype("float32") / base["baseline"].values
        ).astype("float32")

        # Clamp and fill NA for first slots / missing baselines
        s = out[self.name].astype("Float32")  # allow pd.NA
        s = s.clip(upper=50).fillna(0.0)
        out[self.name] = s.astype("float32")
        return out
