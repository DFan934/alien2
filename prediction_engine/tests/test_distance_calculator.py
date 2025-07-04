# ---------------------------------------------------------------------------
# prediction_engine/tests/test_distance_calculator.py
# ---------------------------------------------------------------------------
"""Validate batch_min_k vs brute force"""
import numpy as np
from prediction_engine.distance_calculator import DistanceCalculator

def test_min_k_matches_bruteforce():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(100, 6))
    q   = rng.normal(size=(10, 6))

    dc  = DistanceCalculator(ref, metric="euclidean")
    idx_k, dist_k = dc.batch_min_k(q, k=3)

    # brute‑force baseline
    for r, idx_row in enumerate(idx_k):
        brute = np.argsort(((ref - q[r])**2).sum(1))[:3]
        assert np.array_equal(idx_row, brute)
