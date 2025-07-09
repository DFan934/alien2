# ============================================================================
# FILE: tests/test_distance_cache.py  (NEW)
# ============================================================================
"""Property test – distinct Mahalanobis references must yield distinct Σ⁻¹."""
import hashlib

import numpy as np
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st

from prediction_engine.distance_calculator import _inv_cov_cached


@given(
    arr1=arrays(np.float32, (100, 4), elements=st.floats(-1, 1)),
    arr2=arrays(np.float32, (120, 4), elements=st.floats(-1, 1)),
)
@settings(deadline=None, max_examples=25)
def test_inverse_cov_cache_isolation(arr1: np.ndarray, arr2: np.ndarray):
    # Skip degenerate cases where arrays are all zeros or are equal
    if (
        np.allclose(arr1, 0)
        or np.allclose(arr2, 0)
        or (arr1.shape == arr2.shape and np.allclose(arr1, arr2))
    ):
        return  # degenerate – extremely unlikely
    for arr in (arr1, arr2):
        arr[:, 0] += 1e-3  # ensure non‑singular
    key1 = f"{arr1.shape}-{hashlib.sha1(arr1.tobytes()).hexdigest()}"
    key2 = f"{arr2.shape}-{hashlib.sha1(arr2.tobytes()).hexdigest()}"
    inv1 = _inv_cov_cached(key1, 1e-12, arr1.shape[1], arr1.tobytes())
    inv2 = _inv_cov_cached(key2, 1e-12, arr2.shape[1], arr2.tobytes())
    assert not np.allclose(inv1, inv2)
