############################
# feature_engineering/tests/test_calculators.py
############################
"""Minimal tests to assert shape & NaN‑freeness."""
import pandas as pd
from feature_engineering.calculators.vwap import VWAPCalculator


def test_vwap_calculator_basic():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01 09:30", periods=5, freq="1min", tz="UTC"),
            "close": [10, 11, 12, 13, 14],
            "volume": [100, 200, 300, 400, 500],
        }
    )
    out = VWAPCalculator().transform(df)
    assert not out.isna().any().any()
    assert out.shape == (5, 1)