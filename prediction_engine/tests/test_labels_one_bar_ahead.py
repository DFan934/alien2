import numpy as np
import pandas as pd
from feature_engineering.labels.labeler import one_bar_ahead

def test_one_bar_ahead_alignment():
    df = pd.DataFrame({
        "open": [100, 101, 102, 103, 104],
        "timestamp": pd.date_range("2020-01-01", periods=5, freq="T")
    })
    y = one_bar_ahead(df, horizon=1)
    exp = np.log(np.array([101,102,103,104,104]) / np.array([100,101,102,103,104]))
    # last is NaN by design
    assert np.allclose(y[:-1].to_numpy(), exp[:-1], atol=1e-12)
    assert np.isnan(y.iloc[-1])
