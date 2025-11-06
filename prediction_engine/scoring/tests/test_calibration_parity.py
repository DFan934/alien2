# tests/test_step4_6_calibration_parity.py
import numpy as np
import pandas as pd
from prediction_engine.scoring.batch import score_batch, score_per_symbol_loop

class CalibEV:
    def predict_proba(self, X):  # deterministic
        X = np.asarray(X, float)
        p = 1/(1+np.exp(-(X.sum(axis=1))))
        return p
def calibrate(p):  # same function used in both modes (placeholder)
    # plug your real mapping here if it's pure and deterministic
    return p**0.8 * (1 - p)**0.2 + 0.05

def test_calibrated_probs_identical():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(64, 5))
    ev = CalibEV()
    pv = calibrate(score_batch(ev, X))
    pl = calibrate(score_per_symbol_loop(ev, [X[i] for i in range(len(X))]))
    np.testing.assert_allclose(pv, pl, rtol=0, atol=1e-15)
