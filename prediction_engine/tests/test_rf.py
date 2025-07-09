import numpy as np
from prediction_engine.distance_calculator import DistanceCalculator

def test_rf_metric_weighting():
    rng  = np.random.default_rng(0)
    ref  = rng.normal(size=(200, 32)).astype(np.float32)
    w    = rng.uniform(0.3, 2.0, size=32).astype(np.float32)
    calc = DistanceCalculator(ref, metric="rf_weighted", rf_weights=w)

    x = rng.normal(size=32).astype(np.float32)
    d_calc, idx = calc.top_k(x, 10)

    # manual ground truth for those 10 neighbours
    d_manual = np.array([
        np.sum((x - ref[i])**2 * w) for i in idx
    ], dtype=np.float32)

    np.testing.assert_allclose(d_calc, d_manual, rtol=1e-5)
