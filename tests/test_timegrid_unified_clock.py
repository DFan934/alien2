import pandas as pd
import pytest

from feature_engineering.utils.timegrid import (
    build_unified_clock,
    compute_clock_hash,
    assert_df_on_clock,
    ClockMismatchError,
)


def _bars():
    # Two symbols, 3 minutes total; symbol B missing middle minute.
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "1998-01-01 09:30:10Z",
                    "1998-01-01 09:31:10Z",
                    "1998-01-01 09:32:10Z",
                    "1998-01-01 09:30:40Z",
                    "1998-01-01 09:32:40Z",
                ],
                utc=True,
            ),
            "symbol": ["A", "A", "A", "B", "B"],
            "close": [1, 1, 1, 2, 2],
            "bar_present": [1, 1, 1, 1, 1],
        }
    )


def test_unified_clock_union_observed():
    bars = _bars()
    clock, meta = build_unified_clock(bars, policy="union_observed", min_symbols=1, freq="60s")
    assert len(clock) == 3
    assert meta["policy"] == "union_observed"


def test_unified_clock_min_symbols_observed():
    bars = _bars()
    clock, meta = build_unified_clock(bars, policy="min_symbols_observed", min_symbols=2, freq="60s")
    # only minutes where both A and B present: 09:30 and 09:32 => 2 minutes
    assert len(clock) == 2
    assert meta["policy"] == "min_symbols_observed"
    assert meta["min_symbols"] == 2


def test_clock_hash_stable():
    bars = _bars()
    clock1, _ = build_unified_clock(bars, policy="union_observed", freq="60s")
    clock2, _ = build_unified_clock(bars, policy="union_observed", freq="60s")
    assert compute_clock_hash(clock1) == compute_clock_hash(clock2)


def test_assert_df_on_clock_rejects_extra_ts():
    bars = _bars()
    clock, _ = build_unified_clock(bars, policy="union_observed", freq="60s")
    h = compute_clock_hash(clock)

    df_ok = bars.copy()
    df_ok["timestamp"] = pd.to_datetime(df_ok["timestamp"], utc=True).dt.floor("min")
    assert_df_on_clock(df_ok, clock_index=clock, expected_clock_hash=h, who="ok")

    df_bad = df_ok.copy()
    df_bad.loc[len(df_bad)] = [pd.Timestamp("1998-01-01 10:00:00Z"), "A", 1, 1]
    with pytest.raises(ClockMismatchError):
        assert_df_on_clock(df_bad, clock_index=clock, expected_clock_hash=h, who="bad")
