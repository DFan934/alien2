#=================================
# FILE: prediction_engine/tests/test_ev_engine_meta.py
#======================================

import numpy as np
import types
from prediction_engine.ev_engine import EVEngine, EVResult

def _make_dummy_engine(lambda_reg=0.05, with_meta=False):
    # — minimal, synthetic centroids & stats —
    centers = np.array([[0., 0.], [1., 1.]], dtype=np.float32)
    mu      = np.array([0.01, 0.05], dtype=np.float32)
    var     = np.array([0.10, 0.12], dtype=np.float32)
    var_dn  = np.array([0.08, 0.11], dtype=np.float32)
    eng = EVEngine(
        centers=centers, mu=mu, var=var, var_down=var_dn,
        h=0.5, lambda_reg=lambda_reg,
    )
    if with_meta:
        # Fake meta-model that biases result toward +0.20
        class _MockModel:
            def predict(self, X):
                return np.full((X.shape[0],), 0.20, dtype=float)
        eng._meta_model = _MockModel()
    return eng

def test_meta_blend_shifts_mu():
    x = np.array([0.2, 0.2], dtype=np.float32)

    mu_no_meta = _make_dummy_engine(with_meta=False).evaluate(x).mu
    mu_meta    = _make_dummy_engine(with_meta=True ).evaluate(x).mu

    # meta-model pushes expectation upward toward 0.20
    assert mu_meta > mu_no_meta
    assert abs(mu_meta - 0.20) < abs(mu_no_meta - 0.20)

def test_shrinkage_still_holds():
    """Even with meta-model, λ shrinkage should keep μ in plausible bounds."""
    x = np.array([0.9, 0.9], dtype=np.float32)
    eng = _make_dummy_engine(lambda_reg=0.5, with_meta=True)
    res: EVResult = eng.evaluate(x)

    # 0.5 shrinkage ensures μ not way above 0.20
    assert -0.05 < res.mu < 0.20
    # β still sums to 1
    np.testing.assert_allclose(res.beta.sum(), 1.0, rtol=1e-5)
