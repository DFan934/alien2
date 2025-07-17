# --- tests/test_detectors.py (new)
import pandas as pd
import numpy as np
import pytest
import asyncio


from scanner.detectors import build_from_yaml

# generate 1 day of 1‑min bars (390 rows)
n = 390
idx = pd.date_range("2024-01-01 09:30", periods=n, freq="T")
df = pd.DataFrame({
    "open":   10.0,
    "close":  10.0,
    "high":   10.0,
    "low":    10.0,
    "volume": 1_000,
    "prev_close": 10.0,
    "rvol":   1.0,
}, index=idx)

# one artificial bar that SHOULD pass all filters
df.iloc[100, df.columns.get_loc("open")]  = 10.30           # +3 % gap‑up
df.iloc[100, df.columns.get_loc("prev_close")] = 10.0
df.iloc[100, df.columns.get_loc("rvol")]  = 3.0             # RVOL > 2



@pytest.mark.asyncio
async def test_detector_pass_through():
    comp = build_from_yaml()          # default YAML thresholds
    mask = await comp(df)
    pass_ratio = mask.mean()
    # Expect ≤ 2 % pass‑through  (here exactly 1 / 390 ≈ 0.26 %)
    assert pass_ratio <= 0.02, f"Pass‑through {pass_ratio:.3%} exceeds cap"
