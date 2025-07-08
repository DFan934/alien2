# ---------------------------------------------
# tests/test_distance_calculator_property.py
# ---------------------------------------------
"""Property‑based tests to ensure ``batch_top_k`` matches loop over ``top_k``."""
import numpy as np
import pytest

from prediction_engine.distance_calculator import DistanceCalculator


@pytest.mark.parametrize("metric", ["euclidean", "rf_weighted"])
def test_batch_equals_loop(metric):
    np.random.seed(42)
    n_ref, n_feat, n_q, k = 2048, 16, 64, 7
    ref = np.random.randn(n_ref, n_feat).astype(np.float32)
    rf_w = np.random.rand(n_feat).astype(np.float32) if metric == "rf_weighted" else None
    calc = DistanceCalculator(
        ref,
        metric=metric,
        rf_weights=rf_w,
        ann_backend="sklearn",  # deterministic for unit test
    )
    Q = np.random.randn(n_q, n_feat).astype(np.float32)

    dist_b, idx_b = calc.batch_top_k(Q, k)
    for i in range(n_q):
        dist_single, idx_single = calc.top_k(Q[i], k)
        assert np.allclose(sorted(dist_b[i]), sorted(dist_single), atol=1e-6)
        assert set(idx_b[i]) == set(idx_single)
