# ---------------------------------------------------------------------------
# scanner/tests/test_time_align.py
# ---------------------------------------------------------------------------
"""Verifies that time-alignment makes detector output invariant to the
ordering of raw bars.  A shuffled (out-of-order) DataFrame must yield the
same detector mask as an already-sorted DataFrame once both are snapped
to exact 1-minute buckets.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from scanner.utils import time_align_minute
from scanner.detectors import GapDetector


def _make_df():
    """Six bars, each at a unique minute, so index is unique after alignment."""
    now = pd.Timestamp("2025-07-14 14:30:04", tz="UTC")
    idx = pd.date_range(now, periods=6, freq="1min")
    df = pd.DataFrame(
        {
            "open":    [102, 102.2, 102.4, 102.1, 102.0, 101.8],
            "close":   [102.2, 102.4, 102.1, 102.0, 101.8, 101.7],
            "high":    [102.3, 102.5, 102.4, 102.2, 102.1, 101.9],
            "low":     [101.9, 102.0, 102.0, 101.9, 101.7, 101.6],
            "volume":  np.random.randint(5_000, 10_000, size=6),
            "prev_close": [100] * 6,
            "symbol":  ["TEST"] * 6,
        },
        index=idx,
    )
    return df



def _detect_after_align(df):
    df = df.copy()
    # snap every timestamp to the start of its minute
    df.index = df.index.map(time_align_minute)
    df.sort_index(inplace=True)
    mask = GapDetector(pct=0.02)(df)
    return mask


def test_detector_invariance_after_alignment():
    df_sorted = _make_df()
    df_shuffled = df_sorted.sample(frac=1, random_state=42)  # deterministic shuffle

    mask_sorted   = _detect_after_align(df_sorted)
    mask_shuffled = _detect_after_align(df_shuffled)

    # They must be *exactly* identical after alignment
    pd.testing.assert_series_equal(
        mask_sorted.sort_index(), mask_shuffled.sort_index(), check_names=False
    )
