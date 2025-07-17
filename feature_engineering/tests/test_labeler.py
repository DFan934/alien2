import pandas as pd
from feature_engineering.labels.labeler import one_bar_ahead

def test_one_bar_shift():
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=4, freq="T"),
                       "open": [100, 110, 121, 133.1]})
    y = one_bar_ahead(df, horizon=1)
    # expected log‑returns: 10%, 10%, 10%, nan
    assert abs(y.iloc[0] - 0.0953) < 1e-4
    assert y.iloc[-1] != y.iloc[-1]          # NaN
