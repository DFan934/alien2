# ---------------------------------------------------------------------------
# prediction_engine/tests/test_analogue_synth.py
# ---------------------------------------------------------------------------
"""Unit tests for `analogue_synth.synthesise_ev`"""
import numpy as np
from prediction_engine.analogue_synth import synthesise_ev

def test_synthesise_ev_rmse_improvement():
    rng = np.random.default_rng(42)
    centers = rng.normal(size=(5, 4))
    stats   = {i: (rng.normal(), rng.random()) for i in range(5)}
    x       = rng.normal(size=4)

    idx     = np.arange(5)
    mu_avg  = np.mean([stats[i][0] for i in idx])
    var_avg = np.mean([stats[i][1] for i in idx])

    mu_syn, var_syn = synthesise_ev(x, idx, centers, stats)

    # Assert synthetic variance ≤ average variance and µ closer to 0 target
    assert var_syn <= var_avg + 1e-6
    assert abs(mu_syn) <= abs(mu_avg)