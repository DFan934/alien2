from __future__ import annotations

import pandas as pd
import pytest

from feature_engineering.utils.parquet_utc import (
    TimestampDtypeError,
    read_parquet_utc,
    write_parquet_utc,
)


def test_roundtrip_parquet_preserves_tz_aware_utc(tmp_path):
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["1999-01-05 09:30:00", "1999-01-05 09:31:00"], utc=True),
            "symbol": ["RRC", "RRC"],
            "p": [0.1, 0.2],
        }
    )
    p = tmp_path / "x.parquet"
    write_parquet_utc(df, p, timestamp_cols=("timestamp",))
    out = read_parquet_utc(p, timestamp_cols=("timestamp",), strict=True)

    assert str(out["timestamp"].dtype) == "datetime64[ns, UTC]"


def test_reader_hard_fails_on_tz_naive_datetime64ns(tmp_path):
    # Write a tz-naive parquet deliberately (simulates the exact bug Task 6 is preventing)
    df_bad = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["1999-01-05 09:30:00", "1999-01-05 09:31:00"]),  # naive
            "symbol": ["RRC", "RRC"],
        }
    )
    p = tmp_path / "bad.parquet"
    df_bad.to_parquet(p, index=False)

    with pytest.raises(TimestampDtypeError):
        read_parquet_utc(p, timestamp_cols=("timestamp",), strict=True)
