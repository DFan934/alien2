# prediction_engine/index_backends.py
"""Abstract factory that yields KNN back‑ends (BallTree, Faiss, etc.)."""
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from sklearn.neighbors import BallTree

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None  # noqa: N816


class NearestNeighborIndex(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray) -> None: ...

    @abstractmethod
    def kneighbors(self, Q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (indices, distances)."""


class SklearnIndex(NearestNeighborIndex):
    def __init__(self, metric: str = "euclidean") -> None:  # noqa: D401
        self.tree: BallTree | None = None
        self.metric = metric

    def fit(self, X: np.ndarray) -> None:
        self.tree = BallTree(X, metric=self.metric)

    def kneighbors(self, Q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.tree is None:
            raise RuntimeError("Index not fitted")
        dist, idx = self.tree.query(Q, k=k, return_distance=True)
        return idx, dist


class FaissIndex(NearestNeighborIndex):
    def __init__(self, n_dim: int, metric: str = "l2") -> None:  # noqa: D401
        if faiss is None:
            raise ImportError("faiss not installed; pip install faiss-cpu")
        if metric != "l2":
            raise ValueError("FaissIndex currently supports only L2")
        self.index = faiss.IndexHNSWFlat(n_dim, 32)

    def fit(self, X: np.ndarray) -> None:
        self.index.add(X.astype(np.float32))

    def kneighbors(self, Q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        dist, idx = self.index.search(Q.astype(np.float32), k)
        return idx, dist
