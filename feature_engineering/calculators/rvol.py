############################
# feature_engineering/calculators/rvol.py
############################
"""Relative volume: current volume vs N-day mean for same minute-slot index."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import RollingCalculatorMixin, BaseCalculator
from ..utils.calendar import session_id, slots_since_open


class RVOLCalculator(RollingCalculatorMixin, BaseCalculator):
    def __init__(self, lookback_days: int = 20):
        self.name = f"rvol_{lookback_days}d"
        # 390 min per session (US equities) → store sessions for window calc
        self._days = int(lookback_days)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not {"timestamp", "symbol", "volume"}.issubset(df.columns):
            raise KeyError("RVOLCalculator requires timestamp, symbol, volume")

        # Task 2: canonical grid is fixed at 60s UTC. FE calculators must not infer cadence.
        bar_seconds = 60

        # Slot index within RTH session (0..N-1), cadence-aware
        slot = slots_since_open(df["timestamp"], bar_seconds=bar_seconds)  # int32, -1 outside RTH
        sess = session_id(df["timestamp"])

        out = pd.DataFrame(index=df.index)
        out[self.name] = np.float32(0.0)

        mask = (slot >= 0)
        if not bool(mask.any()):
            return out

        # Build a working frame for only RTH rows, preserving original row order.
        base = df.loc[mask, ["symbol", "volume"]].copy()
        base["_row"] = np.arange(len(base), dtype=np.int64)  # row id in masked frame
        base["slot"] = slot.loc[mask].to_numpy()
        base["sess"] = sess.loc[mask].to_numpy()

        # Ensure numeric volume
        base["volume"] = pd.to_numeric(base["volume"], errors="coerce").astype("float64")

        # Sort for stable rolling by session order per (symbol, slot)
        base = base.sort_values(["symbol", "slot", "sess"], kind="mergesort")

        # Rolling mean across last N sessions per (symbol, slot)
        baseline = (
            base.groupby(["symbol", "slot"], sort=False)["volume"]
                .transform(lambda x: x.rolling(self._days, min_periods=1).mean())
                .astype("float64")
        )

        # Replace non-positive baseline with NaN to avoid inf blowups
        baseline = baseline.mask(baseline <= 0.0, np.nan)

        # Unsort back to the masked-row order so values align 1:1 with df.loc[mask]
        base["baseline"] = baseline.to_numpy()
        base = base.sort_values("_row", kind="mergesort")

        vol = base["volume"].to_numpy(dtype="float64")
        base_bl = base["baseline"].to_numpy(dtype="float64")

        raw = vol / base_bl
        raw[~np.isfinite(raw)] = np.nan
        raw = np.clip(raw, a_min=0.0, a_max=50.0)
        raw = np.nan_to_num(raw, nan=0.0).astype("float32")

        out.loc[mask, self.name] = raw
        out[self.name] = out[self.name].astype("float32")
        return out
