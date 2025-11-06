# tests/test_step4_6_ordering_stable.py
import pandas as pd
from prediction_engine.scoring.batch import vectorize_minute_batch

def test_minute_batch_ordering_stable():
    ts = pd.Timestamp("1999-01-05T09:30Z")
    df = pd.DataFrame({
        "timestamp": [ts, ts, ts, ts],
        "symbol": ["BBB","AAA","DDD","CCC"],  # shuffled
        "pca_0": [0.1, 0.2, 0.3, 0.4],
        "pca_1": [1.0, 1.1, 1.2, 1.3],
    })
    class UnitEV:
        def predict_proba(self, X):  # monotone in rows
            import numpy as np
            return X.sum(axis=1)

    out = vectorize_minute_batch(UnitEV(), df, ["pca_0","pca_1"]).frame
    # Ensure output rows keep the input row order (timestamp, symbol stable)
    assert list(out["symbol"]) == ["BBB","AAA","DDD","CCC"]
