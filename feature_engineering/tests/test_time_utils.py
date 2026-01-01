import pandas as pd
import numpy as np

from feature_engineering.utils.time import to_utc, to_utc_floor, ensure_utc_timestamp_col


def test_to_utc_series_naive_becomes_aware():
    ts = pd.date_range("2020-01-01 09:30", periods=5, freq="T")
    s = pd.Series(ts)  # tz-naive
    out = to_utc(s)
    assert str(out.dtype) == "datetime64[ns, UTC]"


def test_to_utc_series_aware_stays_aware():
    ts = pd.date_range("2020-01-01 09:30", periods=5, freq="T", tz="UTC")
    s = pd.Series(ts)
    out = to_utc(s)
    assert str(out.dtype) == "datetime64[ns, UTC]"


def test_ensure_utc_timestamp_col_enforces_dtype():
    df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=3, freq="T")})
    ensure_utc_timestamp_col(df, "timestamp", who="[TEST]")
    assert str(df["timestamp"].dtype) == "datetime64[ns, UTC]"


def test_to_utc_floor_minute():
    ts = pd.to_datetime(["2020-01-01 09:30:15", "2020-01-01 09:31:59"])
    out = to_utc_floor(ts, "min")
    assert str(out.dtype) == "datetime64[ns, UTC]"
    assert out[0] == pd.Timestamp("2020-01-01 09:30:00+00:00")
