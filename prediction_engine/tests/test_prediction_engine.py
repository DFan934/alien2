# ---------------------------------------------------------------------------
# tests/test_prediction_engine.py
# ---------------------------------------------------------------------------

import math
import numpy as np
import pytest

from prediction_engine.distance_calculator import DistanceCalculator
from prediction_engine.analogue_synth import AnalogueSynth
from prediction_engine.ev_engine import EVEngine


@pytest.mark.parametrize("metric", ["euclidean", "mahalanobis"])
def test_distance_calculator_finite(metric):
    """Ensure DistanceCalculator returns finite distances for both metrics."""
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(10, 5)).astype(np.float32)
    dc = DistanceCalculator(ref, metric=metric)
    x = rng.normal(size=(5,)).astype(np.float32)
    dists, idx = dc(x, k=3)
    assert np.all(np.isfinite(dists)), "Distances contain NaNs or infs"
    assert idx.shape == (3,), "Index output has wrong shape"


def test_mahalanobis_matches_pdist():
    """Mahalanobis distance should match SciPy's pdist for a tiny sample."""
    try:
        from scipy.spatial.distance import mahalanobis as scipy_mah  # type: ignore
    except ImportError:
        pytest.skip("SciPy not available")

    rng = np.random.default_rng(1)
    ref = rng.uniform(-1, 1, size=(4, 4)).astype(np.float32)
    x = ref[0]
    VI = np.linalg.inv(np.cov(ref.T))
    expected = np.array([scipy_mah(x, ref[i], VI) for i in range(len(ref))])
    dc = DistanceCalculator(ref, metric="mahalanobis")
    got, _ = dc(x, k=len(ref))
    # DistanceCalculator sorts by distance; we sort expected likewise
    assert np.allclose(np.sort(got), np.sort(expected), rtol=1e-5, atol=1e-6)


def test_beta_stability_uniform_fallback():
    """AnalogueSynth should fall back to uniform β when NNLS yields zeros."""
    # Construct degenerate ΔX where NNLS likely drives β to zero
    dX = np.zeros((3, 4), dtype=np.float32)
    dy = np.zeros(3, dtype=np.float32)
    beta = AnalogueSynth.solve(dX, dy)
    assert math.isclose(beta.sum(), 1.0, rel_tol=1e-6), "β not renormalised to 1"
    assert np.all(beta >= 0), "β contains negative weights"


def test_ev_engine_returns_finite():
    """EVEngine.evaluate must never return NaNs."""
    rng = np.random.default_rng(42)
    centers = rng.normal(size=(8, 5)).astype(np.float32)
    mu = rng.normal(scale=0.01, size=8).astype(np.float32)
    var = np.full(8, 0.0001, dtype=np.float32)
    downs = var / 2

    ev = EVEngine(
        centers=centers,
        outcome_mu=mu,
        outcome_var=var,
        outcome_var_down=downs,
        h=0.1,
        k=4,
        metric="euclidean",
    )

    x = rng.normal(size=(5,)).astype(np.float32)
    res = ev.evaluate(x, adv=1e7, symbol="TEST")
    assert np.isfinite(res.mu), "Expected value is NaN"
    assert np.isfinite(res.variance), "Variance is NaN"
    assert np.isfinite(res.variance_down), "Downside variance is NaN"
