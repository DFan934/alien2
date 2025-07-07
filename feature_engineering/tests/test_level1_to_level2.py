# ──────────────────────────────────────────────
# tests/test_level1_to_level2.py
# ──────────────────────────────────────────────
import pandas as pd
import numpy as np
from feature_engineering.calculators.multi_tf_trend import MultiTFTrendCalculator
from feature_engineering.calculators.momentum_vol_dynamics import (
    MomentumVolatilityDynamicsCalculator,
)
from feature_engineering.calculators.contextual_zscore import ContextualZScoreCalculator


def _sample_ohlc(n=100):
    idx = pd.date_range("2025-01-01", periods=n, freq="1T")
    return pd.DataFrame(
        {
            "open": np.linspace(100, 110, n),
            "high": np.linspace(101, 111, n),
            "low": np.linspace(99, 109, n),
            "close": np.linspace(100.5, 110.5, n),
        },
        index=idx,
    )


def test_multi_tf_trend():
    df = _sample_ohlc(120)
    calc = MultiTFTrendCalculator(intervals=[2, 4])
    out = calc(df)
    assert out.filter(like="trend_vs_indecision_2m").notna().any().all()
    assert out.filter(like="fakeout_count_4m").notna().any().all()


def test_momvol_nan_free():
    """Only requirement: the most‑recent bar (used in live inference) must be finite."""
    df = _sample_ohlc(20)
    calc = MomentumVolatilityDynamicsCalculator(window=4)
    res = calc(df)
    # Latest row should be NaN‑free
    assert np.isfinite(res.iloc[-1]).all()

def test_zscore_has_columns():
    df = _sample_ohlc(300)
    res = ContextualZScoreCalculator()(df)
    assert {"z_body", "z_wick"}.issubset(res.columns)
