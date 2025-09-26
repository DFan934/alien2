import pandas as pd
import numpy as np
from feature_engineering.labels.labeler import one_bar_ahead, one_bar_ahead_binary

def _mk_df(seq):
    # minimal frame with ascending timestamps & open column
    ts = pd.date_range("2020-01-01", periods=len(seq), freq="T")
    return pd.DataFrame({"timestamp": ts, "open": np.array(seq, dtype=float)})

def test_rising_series_positive_label():
    df = _mk_df([100, 101, 102, 103, 104, 105])
    r = one_bar_ahead(df, horizon=2)             # log(open[t+2]/open[t])
    y = one_bar_ahead_binary(df, horizon=2)
    assert r.iloc[0] > 0 and y.iloc[0] == 1
    assert r.iloc[-1] != r.iloc[-1]              # NaN for final 2 rows
    assert y.iloc[-1] != y.iloc[-1]

def test_falling_series_negative_label():
    df = _mk_df([105, 104, 103, 102, 101, 100])
    r = one_bar_ahead(df, horizon=3)
    y = one_bar_ahead_binary(df, horizon=3)
    assert r.iloc[0] < 0 and y.iloc[0] == 0
