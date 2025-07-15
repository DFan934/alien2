# ============================================================================
# FILE: tests/test_analogue_synth.py  (NEW)
# ============================================================================
import numpy as np
import pytest

from prediction_engine.analogue_synth import AnalogueSynth
from prediction_engine.ev_engine import EVEngine, EVResult


def test_beta_non_trivial():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(50, 6)).astype(np.float32)
    mu = rng.normal(scale=0.01, size=50).astype(np.float32)
    var = np.full(50, 0.02, dtype=np.float32)
    var_down = var.copy()

    engine = EVEngine(
        centers=centers,
        mu=mu,
        var=var,
        var_down=var_down,
        h=0.5,
    )
    x = rng.normal(size=6).astype(np.float32)
    res: EVResult = engine.evaluate(x)

    assert (res.beta > 0).sum() > 0, "Beta should contain positive weights"




def test_exact_cancellation_1d():
    # Two neighbours at +1 and -1 around x0=0 should split 50/50
    delta_mat = np.array([[1.0], [-1.0]], dtype=float)
    target_delta = np.array([0.0], dtype=float)
    beta, residual = AnalogueSynth.weights(delta_mat, target_delta)
    # both weights ≈ 0.5, and perfect fit residual
    assert beta.shape == (2,)
    assert pytest.approx(0.5, rel=1e-6) == beta[0]
    assert pytest.approx(0.5, rel=1e-6) == beta[1]
    assert residual == pytest.approx(0.0, abs=1e-8)

    mu_nn = np.array([0.123, 0.123])
    beta, residual = AnalogueSynth.weights(delta_mat, target_delta)
    mu_syn = beta @ mu_nn
    assert mu_syn == pytest.approx(mu_nn[0])


def test_uniform_fallback_on_zero_deltas():
    # If delta_mat is all zeros, we fall back to uniform across k neighbours
    k, d = 4, 3
    delta_mat = np.zeros((k, d), dtype=float)
    target_delta = np.zeros(d, dtype=float)
    beta, residual = AnalogueSynth.weights(delta_mat, target_delta)
    assert beta.shape == (k,)
    for w in beta:
        assert w == pytest.approx(1.0 / k, rel=1e-6)
    assert residual == pytest.approx(0.0, abs=1e-8)

def test_variance_weighting_prefers_low_variance():
    # Given two neighbours symmetrically placed,
    # the one with lower var_nn should get a higher β.
    delta_mat = np.array([[1.0], [-1.0]], dtype=float)
    target_delta = np.array([0.0], dtype=float)
    var_nn = np.array([1.0, 100.0], dtype=float)  # neighbour 0 is 10× less risky
    beta, residual = AnalogueSynth.weights(delta_mat, target_delta, var_nn=var_nn)
    assert beta[0] > beta[1]
    assert beta.sum() == pytest.approx(1.0, rel=1e-6)
    # residual should be non-negative (we emphatically check weight ordering above)
    assert residual >= 0.0
