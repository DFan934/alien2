import pandas as pd
import pytest

from feature_engineering.utils.timegrid import (
    compute_clock_hash,
    assert_df_on_clock,
    ClockMismatchError,
)


def test_prediction_engine_stage_must_use_unified_clock():
    clock = pd.date_range("1998-01-01 09:30:00Z", periods=3, freq="1min")
    h = compute_clock_hash(clock)

    decisions = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["1998-01-01 09:30:00Z", "1998-01-01 09:31:00Z"], utc=True
            ),
            "symbol": ["A", "B"],
            "target_qty": [10, 0],
        }
    )
    assert_df_on_clock(decisions, clock_index=clock, expected_clock_hash=h, who="decisions_ok")

    decisions_bad = decisions.copy()
    decisions_bad.loc[len(decisions_bad)] = ["1998-01-01 09:40:00Z", "A", 5]
    with pytest.raises(ClockMismatchError):
        assert_df_on_clock(decisions_bad, clock_index=clock, expected_clock_hash=h, who="decisions_bad")
