# ============================================================================
# FILE: prediction_engine/distance_calculator.py
# UPDATED: fixed Mahalanobis cache‑key, auto‑load RF weights artefact
# ============================================================================
"""Fast *k*‑NN distance lookup with FAISS / BallTree back‑ends and RF‑weights.

Key changes in this patch
------------------------
* Mahalanobis inverse‑covariance matrix is now cached by **shape + SHA‑1 hash** of
  the reference bytes instead of the ambiguous `id(ref)` heuristic – eliminates
  cross‑contamination when two matrices share column count.
* `DistanceCalculator.from_artifacts()` automatically looks for
  `rf_feature_weights.npy` alongside the reference file when the requested
  metric is ``"rf_weighted"``.
* Added tighter runtime shape checks and clarified error paths.
"""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path
import time
from typing import Any, Literal, Tuple

import numpy as np
from numpy.typing import NDArray

from .index_backends import get_index, NearestNeighborIndex

Metric = Literal["euclidean", "mahalanobis", "rf_weighted"]


# ---------------------------------------------------------------------------
# Distance helper (Mahalanobis) – cached per (id(ref), eps)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=128)
def _inv_cov_cached(
    key: str,  # "(rows, cols)-sha1"  – ensures uniqueness per matrix
    eps: float,
    n_features: int,
    ref_bytes: bytes,
) -> NDArray[np.float32]:
    """Return (Σ + eps·I)^‑¹ for the reference array encoded in *ref_bytes*."""
    arr = np.frombuffer(ref_bytes, dtype=np.float32).reshape(-1, n_features)
    cov = np.cov(arr, rowvar=False, dtype=np.float64)
    cov.flat[:: cov.shape[0] + 1] += eps  # Tikhonov regularisation
    return np.linalg.inv(cov).astype(np.float32)


# ---------------------------------------------------------------------------
# Main calculator
# ---------------------------------------------------------------------------
class DistanceCalculator:  # pylint: disable=too-few-public-methods
    """Vectorised K‑NN search against a fixed reference matrix."""

    __slots__ = (
        "_ref",
        "_metric",
        "_inv_cov",
        "_ann_backend",
        "_index",
        "_rf_w",
        "_recency_w",
    )

    def __init__(
        self,
        ref: NDArray[np.float32],
        *,
        metric: Metric = "euclidean",
        rf_weights: NDArray[np.float32] | None = None,
        ann_backend: str = "sklearn",
        eps: float = 1e-12,
        backend_kwargs: dict[str, Any] | None = None,
        recency_weights: NDArray[np.float32] | None = None,
    ) -> None:
        if ref.ndim != 2:
            raise ValueError("ref must be a 2‑D array [n_reference, n_features]")
        self._ref = np.ascontiguousarray(ref.astype(np.float32))
        self._metric: Metric = metric  # type: ignore[assignment]
        self._ann_backend = ann_backend.lower()
        backend_kwargs = backend_kwargs or {}

        # ------------------------------------------------------------------
        # Build ANN index when: (metric == euclidean)  AND  backend != NONE
        # ------------------------------------------------------------------
        if metric == "euclidean":
            self._index: NearestNeighborIndex | None = get_index(
                backend=self._ann_backend,
                n_dim=self._ref.shape[1],
                metric="euclidean",
                **backend_kwargs,
            )
            self._index.fit(self._ref)
        else:
            self._index = None

        # ------------------------------------------------------------------
        # Mahalanobis pre‑compute
        # ------------------------------------------------------------------
        if metric == "mahalanobis":
            byt = self._ref.tobytes()
            sha = hashlib.sha1(byt).hexdigest()
            #key = f"{self._ref.shape}-{int(time.time())}-{sha}"
            key = f"{self._ref.shape}-{sha}"
            self._inv_cov = _inv_cov_cached(key, eps, self._ref.shape[1], byt)
        else:
            self._inv_cov = None  # type: ignore[assignment]

        # ------------------------------------------------------------------
        # RF‑weighted pre‑multiplied reference (diag(weight) · ref)
        # ------------------------------------------------------------------
        if metric == "rf_weighted":
            if rf_weights is None:
                raise ValueError("rf_weighted metric requires rf_weights vector")
            self._rf_w = np.asarray(rf_weights, dtype=np.float32)
            if self._rf_w.ndim != 1 or self._rf_w.shape[0] != self._ref.shape[1]:
                raise ValueError("rf_weights shape mismatch – must be (n_features,)")
            # NOTE: self._ref stays *unchanged*  (no double-weighting)
        else:
            self._rf_w = None
            # optional recency curve weights (same length as n_features)
        self._recency_w = np.asarray(recency_weights,
        dtype=np.float32) if recency_weights is not None else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def top_k(self, x: NDArray[np.float32], k: int) -> Tuple[NDArray[np.float32], NDArray[np.int32]]:
        """Return (dists, idxs) for a single query vector."""
        d, i = self.batch_top_k(x[np.newaxis, :], k)
        return d[0], i[0]

    # In prediction_engine/distance_calculator.py

    def batch_top_k(
            self, Q: NDArray[np.float32], k: int, *, workers: int = 0
    ) -> Tuple[NDArray[np.float32], NDArray[np.int32]]:
        """Vectorised top-k for a batch of queries Q (n_queries, n_features)."""

        print("--- EXECUTING CORRECT BATCH_TOP_K METHOD ---")

        if Q.ndim != 2:
            raise ValueError("Q must be 2-D")
        if not 0 < k <= self._ref.shape[0]:
            raise ValueError("Invalid k value")

        # FIX: The "fast path" now correctly handles the (indices, distances)
        # signature returned from the backend indexer.
        #if self._index is not None:
        if self._index is not None and self._recency_w is None:
            # 1. Unpack correctly: indices first, then squared distances.
            indices, squared_distances = self._index.kneighbors(Q, k)
            # 2. Return in the consistent (distances, indices) order.
            return squared_distances, indices

        # Fallback for other metrics
        if self._metric == "mahalanobis":
            diff = Q[:, None, :] - self._ref[None, :, :]
            left = diff @ self._inv_cov
            dist2 = np.sum(left * diff, axis=2, dtype=np.float32)
        elif self._metric == "rf_weighted":
            diff = Q[:, None, :] - self._ref[None, :, :]
            dist2 = np.sum(diff * diff * self._rf_w, axis=2, dtype=np.float32)
        else:  # Default to Euclidean if no indexer is used
            #diff = Q[:, None, :] - self._ref[None, :, :]
            #dist2 = np.sum(diff * diff, axis=2, dtype=np.float32)
            diff = Q[:, None, :] - self._ref[None, :, :]
            # apply recency weights if provided
            if self._recency_w is not None:
                diff = diff * self._recency_w  # broadcast across rows
            dist2 = np.sum(diff * diff, axis=2, dtype=np.float32)
        # Partition and sort to get the top k results
        idx = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
        part = np.take_along_axis(dist2, idx, axis=1)
        order = np.argsort(part, axis=1)
        sorted_idx = np.take_along_axis(idx, order, axis=1)
        sorted_dist2 = np.take_along_axis(part, order, axis=1)

        return sorted_dist2, sorted_idx
    # ------------------------------------------------------------------
    # Convenience ctor – from artefact folder
    # ------------------------------------------------------------------
    @classmethod
    def from_artifacts(
        cls,
        ref_path: str | Path,
        *,
        metric: Metric = "euclidean",
        rf_weights_path: str | None = None,
        ann_backend: str = "sklearn",
        backend_kwargs: dict[str, Any] | None = None,
    ) -> "DistanceCalculator":
        ref_arr = np.load(Path(ref_path), mmap_mode="r")

        # Auto‑discover RF weights if needed and caller did not specify a path
        if metric == "rf_weighted" and rf_weights_path is None:
            for fname in ("rf_feature_weights.npy", "weights.npy"):
                cand = Path(ref_path).with_name(fname)
                if cand.exists():
                    rf_weights_path = cand
                    break

        #w = np.load(rf_weights_path, allow_pickle=False) if rf_weights_path else None

        # inside DistanceCalculator.from_artifacts(...)
        w = np.load(rf_weights_path, allow_pickle=False) if rf_weights_path else None
        if metric == "rf_weighted" and w is not None:
            w = w.astype(np.float32, copy=False)
            s = float(w.sum())
            if s <= 0 or not np.isfinite(s):
                raise ValueError("rf_feature_weights sums to non-positive")
            w /= s

        return cls(
            ref_arr,
            metric=metric,
            rf_weights=w,
            ann_backend=ann_backend,
            backend_kwargs=backend_kwargs,
        )

    # ------------------------------------------------------------------
    # Callable – keep legacy signature ``idxs, dist2``
    # ------------------------------------------------------------------

    def __call__(self, x: NDArray[np.float32], k: int) -> Tuple[NDArray[np.int32], NDArray[np.float32]]:
        # self.top_k returns (distances, indices)
        distances, indices = self.top_k(x, k)
        # Return in the documented order: (indices, distances)
        return indices, distances

