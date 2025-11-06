import numpy as np
import pandas as pd
from prediction_engine.scoring.batch import score_batch, score_per_symbol_loop, vectorize_minute_batch

class FakeEV:
    # deterministic linear probability with sigmoid; no randomness
    def __init__(self, w):
        self.w = np.asarray(w, dtype=float).reshape(-1)
    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        z = X @ self.w
        p = 1.0 / (1.0 + np.exp(-z))
        return p

def test_vectorized_vs_loop_identical_probs():
    rng = np.random.default_rng(42)
    F = 6
    N = 128
    w = rng.normal(0, 0.5, size=F)
    X = rng.normal(0, 1, size=(N, F))

    ev = FakeEV(w)
    pv = score_batch(ev, X)
    pl = score_per_symbol_loop(ev, [X[i] for i in range(N)])
    #np.testing.assert_allclose(pv, pl, rtol=0, atol=0)
    np.testing.assert_allclose(pv, pl, rtol=0, atol=1e-15)


def test_vectorize_minute_batch_dataframe_shape_and_cols():
    rng = np.random.default_rng(0)
    F = 4
    N = 10
    w = rng.normal(0, 0.5, size=F)
    ev = FakeEV(w)

    ts = pd.Timestamp("1999-01-05 09:30Z")
    df = pd.DataFrame({
        "timestamp": [ts]*N,
        "symbol": [f"S{i:02d}" for i in range(N)],
        **{f"pca_{i}": rng.normal(size=N) for i in range(F)},
    })
    out = vectorize_minute_batch(ev, df, [f"pca_{i}" for i in range(F)]).frame
    assert list(out.columns) == ["timestamp","symbol","p_raw","p_cal"]
    assert len(out) == N
