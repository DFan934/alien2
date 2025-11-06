import numpy as np
import time
from prediction_engine.scoring.batch import score_batch, score_per_symbol_loop

class SleepyEV:
    """
    Simulates Python-call overhead: vectorized path pays it ONCE, loop pays it per row.
    This yields a robust >=20% speedup even on tiny matrices.
    """
    def __init__(self, sleep_s=0.001):
        self.sleep_s = float(sleep_s)
    def predict_proba(self, X):
        time.sleep(self.sleep_s)  # fixed per-call overhead
        X = np.asarray(X, dtype=float)
        # trivial prob model
        return 1.0 / (1.0 + np.exp(-X.sum(axis=1)))

def test_vectorized_is_faster_by_20_percent():
    N, F = 128, 8
    X = np.ones((N, F), dtype=float) * 0.1
    ev = SleepyEV(sleep_s=0.0005)  # keep CI stable

    t0 = time.perf_counter()
    _ = score_per_symbol_loop(ev, [X[i] for i in range(N)])
    t_loop = time.perf_counter() - t0

    t1 = time.perf_counter()
    _ = score_batch(ev, X)
    t_vec = time.perf_counter() - t1

    # Require at least 20% win
    assert t_vec <= 0.8 * t_loop, f"Expected >=20% speedup, got loop={t_loop:.4f}s vec={t_vec:.4f}s"
