# ============================================================================
# FILE: prediction_engine/index_backends.py
# Re‑written with FAISS support and a tiny factory helper.
# ============================================================================
"""Abstract factory that yields approximate‑nearest‑neighbour (ANN) indices.

Exposes
    * ``SklearnIndex``  – exact BallTree (small data / debug)
    * ``FaissIndex``    – HNSWFlat (CPU) or IVF‑PQ+OPQ (GPU if available)

The user selects the backend via ``EngineConfig.ann_backend``.  The factory
also accepts backend‑specific kwargs so you can tweak, e.g., ``efSearch`` for
HNSW or nlist / nprobe for IVF.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np
from sklearn.neighbors import BallTree

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None  # noqa: N816

__all__ = [
    "NearestNeighborIndex",
    "SklearnIndex",
    "FaissIndex",
    "get_index",
]


class NearestNeighborIndex(ABC):
    """Minimal ANN interface so we can swap implementations transparently."""

    @abstractmethod
    def fit(self, X: np.ndarray) -> None: ...

    @abstractmethod
    def kneighbors(self, Q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (indices, squared_distances)."""


# ---------------------------------------------------------------------------
# Exact KNN – sklearn BallTree (debug / small refs < 50 K)
# ---------------------------------------------------------------------------
class SklearnIndex(NearestNeighborIndex):
    def __init__(self, metric: str = "euclidean", **_: Any):
        self.tree: BallTree | None = None
        self.metric = metric

    def fit(self, X: np.ndarray) -> None:
        self.tree = BallTree(X, metric=self.metric)

    def kneighbors(self, Q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.tree is None:
            raise RuntimeError("Index not fitted")
        dist, idx = self.tree.query(Q, k=k, return_distance=True)
        # sklearn returns *distance*, convert to squared for consistency
        return idx.astype(np.int32), (dist ** 2).astype(np.float32)


# ---------------------------------------------------------------------------
# Fast ANN – FAISS (HNSWFlat CPU, IVF‑PQ (+OPQ) GPU)
# ---------------------------------------------------------------------------
class FaissIndex(NearestNeighborIndex):
    """Thin wrapper around Facebook Faiss.

    *CPU path*  – default to HNSWFlat which performs well for 10 M vectors.
    *GPU path*  – if compiled with GPU, fall back to IVF‑PQ+OPQ for memory.
    """

    def __init__(self, n_dim: int, metric: str = "l2", **faiss_kwargs: Any):  # noqa: D401
        if faiss is None:
            raise ImportError("faiss not installed; pip install faiss-cpu")
        if metric != "l2":
            raise ValueError("FaissIndex currently supports only L2/SSE distance")

        self.metric = metric
        self.n_dim = n_dim
        self.faiss_kwargs = faiss_kwargs

        # --- decide CPU vs GPU ------------------------------------------------
        self._gpu = faiss.get_num_gpus() > 0  # type: ignore[attr-defined]

        if self._gpu:
            nlist = faiss_kwargs.get("nlist", 1024)
            m = faiss_kwargs.get("pq_m", 32)
            quantiser = faiss.IndexFlatL2(n_dim)
            index_ivf = faiss.IndexIVFPQ(quantiser, n_dim, nlist, m, 8)
            index_ivf.opq = faiss.OPQMatrix(n_dim, m)
            self.index = index_ivf
        else:
            hnsw_m = faiss_kwargs.get("hnsw_m", 32)
            self.index = faiss.IndexHNSWFlat(n_dim, hnsw_m)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> None:
        X = np.ascontiguousarray(X.astype(np.float32))
        if self._gpu:
            # move the coarse quantiser to GPU before training
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)  # type: ignore[misc]
        if self.index.is_trained:
            self.index.reset()
        else:
            self.index.train(X)
        self.index.add(X)

    def kneighbors(self, Q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if not self.index.ntotal:
            raise RuntimeError("Index not fitted")
        D, I = self.index.search(np.ascontiguousarray(Q.astype(np.float32)), k)
        return I.astype(np.int32), D.astype(np.float32)  # D already squared


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def get_index(backend: str, n_dim: int, metric: str = "euclidean", **kwargs: Any) -> NearestNeighborIndex:  # noqa: D401
    backend = backend.lower()
    if backend == "sklearn":
        return SklearnIndex(metric=metric)
    if backend == "faiss":
        return FaissIndex(n_dim=n_dim, metric="l2", **kwargs)
    raise ValueError(f"Unknown ANN backend: {backend}")