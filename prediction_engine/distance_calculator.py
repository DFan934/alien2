# ---------------------------------------------------------------------------
# prediction_engine/distance_calculator.py  –  safe argpartition
# ---------------------------------------------------------------------------
"""Fast distance calculator with robust regularisation and caching.
Fixes off‑by‑one in `np.argpartition` when k == n_cols.
"""
from __future__ import annotations

from pathlib import Path

"""High‑performance distance calculator with configurable metric.

Implements Euclidean **and** Mahalanobis distances with an LRU‑style cache that
keys on the *identity* of the reference matrix rather than its full byte
content.  This avoids runaway memory usage when the same reference set is used
millions of times during back‑tests.

This file belongs in  ``prediction_engine/distance_calculator.py``  and is a
*drop‑in* replacement for the previous stub.
"""

import functools
import hashlib
from typing import Literal, Tuple
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

#Metric = Literal["euclidean", "mahalanobis"]
Metric = Literal["euclidean", "mahalanobis", "rf_weighted"]


# ---------------------------------------------------------------------------
# Helper – cached inverse covariance
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers – LRU‑cached inverse covariance
# ---------------------------------------------------------------------------

def _inv_cov_cached(ref_id: int, arr: NDArray[np.float32], eps: float) -> NDArray[np.float32]:
    """Cache inverse covariance per *identity* of the reference array."""
    key = (ref_id, eps)
    if key in _inv_cov_cached._store:  # type: ignore[attr-defined]
        return _inv_cov_cached._store[key]
    cov = np.cov(arr, rowvar=False, dtype=np.float64)
    cov.flat[:: cov.shape[0] + 1] += eps
    inv = np.linalg.inv(cov).astype(np.float32)
    _inv_cov_cached._store[key] = inv
    if len(_inv_cov_cached._store) > 8:  # simple LRU max‑len
        _inv_cov_cached._store.pop(next(iter(_inv_cov_cached._store)))
    return inv


_inv_cov_cached._store: dict[Tuple[int, float], NDArray[np.float32]] = {}  # type: ignore[attr-defined]



# Global weak‑registry so the lru_cache can map id → ndarray
_REF_REGISTRY: dict[int, NDArray[np.float32]] = {}


class DistanceCalculator:  # pylint: disable=too-few-public-methods
    """Vectorised *k*-NN distance lookup with small‑footprint caching.

    Parameters
    ----------
    ref : NDArray[np.float32]
        A 2‑D matrix of shape (n_reference, n_features) in **row‑major** order.
        The matrix is **not** copied; keep it immutable after construction.
    metric : {"euclidean", "mahalanobis"}
        Distance metric to use.  Mahalanobis automatically inverts the
        covariance of *ref* and caches it.
    eps : float, default 1e-12
        Numerical jitter added to the diagonal before matrix inversion to
        prevent singularities.
    """

    __slots__ = ("_ref", "_metric", "_inv_cov","_rf_w", "_cache")

    def __init__(
        self,
        ref: NDArray[np.float32],
        *,
        metric: Metric = "euclidean",
        eps: float = 1e-12,
        cov_inv: NDArray[np.float32] | None = None,  # <-- NEW
        rf_weights: NDArray[np.float32] | None = None,  # <-- NEW
    ) -> None:
        if ref.ndim != 2:
            raise ValueError("ref must be a 2‑D array [n_reference, n_features]")
        self._ref: NDArray[np.float32] = ref.astype(np.float32, copy=False)
        self._metric: Metric = metric  # type: ignore[assignment]
        self._cache: dict[Tuple[int, int], Tuple[NDArray[np.float32], NDArray[np.int64]]] = {}
        #self._rf_w = rf_weights.astype(np.float32) if rf_weights is not None else None


        if metric == "mahalanobis":
            cov = np.cov(self._ref, rowvar=False, dtype=np.float64)
            # jitter for stability
            cov.flat[:: cov.shape[0] + 1] += eps
            self._inv_cov = np.linalg.inv(cov).astype(np.float32)
        else:
            self._inv_cov = None  # type: ignore[assignment]

        # ---- RF‑weighted ----------------------------------------------------
        self._rf_w = rf_weights.astype(np.float32, copy=False) if rf_weights is not None else None
        if metric == "rf_weighted" and self._rf_w is None:
            raise ValueError("rf_weighted metric requires rf_weights vector")


    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def top_k(self, x: NDArray[np.float32], k: int) -> Tuple[NDArray[np.float32], NDArray[np.int64]]:
        """Return the *k* smallest distances and their indices.

        Parameters
        ----------
        x : (n_features,) vector representing the query setup.
        k : number of neighbours to return.
        """
        if x.ndim != 1:
            raise ValueError("x must be a 1‑D feature vector")
        if k <= 0:
            raise ValueError("k must be positive")
        if k > self._ref.shape[0]:
            raise ValueError("k cannot exceed reference size")

        key = self._make_key(x)
        try:
            dists, idxs = self._cache[key]
            return dists[:k], idxs[:k]
        except KeyError:
            pass  # cold‑start

        # Compute full distance vector
        if self._metric == "euclidean":
            diff = self._ref - x
            dist_vec = np.sqrt(np.sum(diff * diff, axis=1, dtype=np.float32))
        elif self._metric == "rf_weighted":
            if self._rf_w is None:
                raise RuntimeError("rf_weighted metric needs rf_weights vector")
            diff = self._ref - x
            dist_vec = np.sqrt(np.sum(self._rf_w * diff * diff, axis=1, dtype=np.float32))
        else:  # Mahalanobis
            diff = self._ref - x
            left = diff @ self._inv_cov
            dist_vec = np.sqrt(np.sum(left * diff, axis=1, dtype=np.float32))

        idxs = np.argpartition(dist_vec, kth=k - 1)[:k]
        dists = dist_vec[idxs]
        # Sort the *k* distances/idxs for deterministic output
        order = np.argsort(dists)
        dists, idxs = dists[order], idxs[order]

        # Insert into cache (simple LRU maxlen=64)
        if len(self._cache) >= 64:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = (dists.copy(), idxs.copy())
        return dists, idxs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_key(self, x: NDArray[np.float32]) -> Tuple[int, int]:
        """Stable 2‑tuple key: id(ref) ^ hash(x‑bytes).

        Using *id(ref)* keeps separate DistanceCalculator instances from
        colliding while avoiding storing the entire reference matrix in the
        cache key (which previously caused memory bloat).  The feature vector
        is hashed via SHA‑1 on its bytes representation, reduced to an int.
        """
        h = hashlib.sha1(x.tobytes()).digest()
        # Use first 8 bytes as int
        x_hash = int.from_bytes(h[:8], "little", signed=False)
        return (id(self._ref), x_hash)

    def __call__(self, x: NDArray[np.float32], k: int):
        """Alias so `EVEngine` can call the instance directly."""
        dists, idxs = self.top_k(x, k)
        return idxs, dists * dists  # EVEngine wants squared distances

    # ------------------------------------------------------------------
    # Convenience constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_artifacts(
            cls,
            ref_path: str | Path,
            *,
            metric: Metric = "euclidean",
            rf_weights: str | None = None,
    ) -> "DistanceCalculator":
        ref_arr = np.load(Path(ref_path), mmap_mode="r")
        w = np.load(rf_weights) if rf_weights else None
        return cls(ref_arr, metric=metric, rf_weights=w)