# tests/test_step4_6_empty_minute_safe.py
import pandas as pd
from scripts.run_backtest import _score_minute_batch_shim

def test_empty_minute_returns_empty_df():
    class EV:
        def predict_proba(self, X): raise AssertionError("should not be called")
    empty = pd.DataFrame(columns=["timestamp","symbol","pca_0"])
    out = _score_minute_batch_shim(EV(), empty)
    assert out.empty
