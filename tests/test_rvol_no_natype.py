import numpy as np
import pandas as pd

from feature_engineering.calculators.rvol import RVOLCalculator


def test_rvol_never_emits_pd_na_or_object():
    ts = pd.date_range("1998-08-26 09:30:00", periods=10, freq="60s", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts,
        "symbol": ["X"] * 10,
        "volume": [100, np.nan, 0, 50, np.nan, 10, 0, 5, np.nan, 1],
        "open": np.linspace(10, 11, 10),
        "high": np.linspace(10, 11, 10),
        "low": np.linspace(10, 11, 10),
        "close": np.linspace(10, 11, 10),
    })

    calc = RVOLCalculator(lookback_days=2)
    out = calc.transform(df)

    assert out.shape[0] == df.shape[0]
    s = out[calc.name]
    assert s.dtype == "float32"
    # no pd.NA should survive
    assert not s.astype("object").eq(pd.NA).any()
