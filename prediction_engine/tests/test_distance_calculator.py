# ---------------------------------------------------------------------------
# prediction_engine/tests/test_distance_calculator.py
# ---------------------------------------------------------------------------
"""Validate batch_min_k vs brute force"""
import numpy as np
from prediction_engine.distance_calculator import DistanceCalculator


# In prediction_engine/tests/test_distance_calculator.py

def test_min_k_matches_bruteforce():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(100, 6))
    q = rng.normal(size=(10, 6))

    dc = DistanceCalculator(ref, metric="euclidean")

    # Unpack the results from the method being tested
    # Note: batch_top_k returns (distances, indices)
    dist_k, idx_k = dc.batch_top_k(q, k=3)

    # Brute-force baseline
    for r, idx_row in enumerate(idx_k):
        # Calculate brute-force distances and get the top 3 indices
        brute_dists = ((ref - q[r]) ** 2).sum(1)
        brute_indices = np.argsort(brute_dists)[:3]

        # THE FIX: Sort both arrays numerically before comparing.
        # This makes the test robust to different tie-breaking orders.
        assert np.array_equal(np.sort(idx_row), np.sort(brute_indices))
