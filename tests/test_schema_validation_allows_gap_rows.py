import pandas as pd
import pytest

from feature_engineering.pipelines.core import _assert_schema_tz_freq

def test_schema_validation_allows_nan_gap_rows():
    df = pd.DataFrame({
        "symbol": ["X", "X"],
        "timestamp": [
            pd.Timestamp("2025-01-01 10:00:00", tz="UTC"),
            pd.Timestamp("2025-01-01 10:01:00", tz="UTC"),
        ],
        # first row is a valid bar
        "open": [10.0, None],
        "high": [10.5, None],
        "low":  [9.8, None],
        "close":[10.2, None],
        # volume can be 0 on missing bars after grid standardization
        "volume": [100.0, 0.0],
    })
    # Should not raise: the 2nd row is an intentional gap row.
    _assert_schema_tz_freq(df)

def test_schema_validation_still_rejects_true_high_violations():
    df = pd.DataFrame({
        "symbol": ["X"],
        "timestamp": [pd.Timestamp("2025-01-01 10:00:00", tz="UTC")],
        "open": [10.0],
        "close": [12.0],
        "high": [11.0],   # invalid: high < max(open, close)=12
        "low":  [9.0],
        "volume": [100.0],
    })
    with pytest.raises(ValueError, match="high must be >="):
        _assert_schema_tz_freq(df)
