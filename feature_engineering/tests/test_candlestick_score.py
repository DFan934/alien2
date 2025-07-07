# ---------------------------------------------------------------------------
# --- tests/test_candlestick_score.py ---------------------------------------
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
from feature_engineering.calculators.candlestick_score import (
    candlestick_trend_vs_indecision,
    TrendVsIndecisionCalculator,
)

def _sample_df(n: int = 10) -> pd.DataFrame:
    rng = pd.date_range("2025-01-01", periods=n, freq="1min")
    return pd.DataFrame(
        {
            "open": np.linspace(100, 109, n),
            "high": np.linspace(101, 110, n),
            "low": np.linspace( 99, 108, n),
            "close": np.linspace(100.5, 109.5, n),
        },
        index=rng,
    )

def test_function_no_nan():
    df = _sample_df(20)
    s = candlestick_trend_vs_indecision(df, window=4)
    assert np.isfinite(s.iloc[-1])


def test_class_wrapper():
    df = _sample_df(15)
    calc = TrendVsIndecisionCalculator()
    s = calc(df)
    assert s.name == "trend_vs_indecision"
    assert len(s) == len(df)
