# ============================================
# File: tests/test_analogue_synth.py  (NEW)
# --------------------------------------------
"""Property tests for analogue_synth.solve."""

import numpy as np
from prediction_engine import analogue_synth as asynth


def test_solve_basic_rmse():
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=8)
    X_nn = x0 + rng.normal(scale=0.1, size=(5, 8))  # close neighbours
    y_nn = rng.normal(loc=0.05, scale=0.02, size=5)

    mu_syn, var_syn, beta = asynth.solve(x0, X_nn, y_nn)

    # Baseline: simple average of neighbour outcomes
    mu_avg = y_nn.mean()
    rmse_syn = np.sqrt((mu_syn - y_nn) ** 2).mean()
    rmse_avg = np.sqrt((mu_avg - y_nn) ** 2).mean()

    assert abs(beta.sum() - 1) < 1e-6
    assert rmse_syn <= rmse_avg  # synthesized mean should not be worse than avg
    assert var_syn >= 0
# ============================================
