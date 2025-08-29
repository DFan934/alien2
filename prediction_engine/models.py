# ---------------------------------------------------------------------------
# FILE: prediction_engine/models.py   (REPLACEMENT)
# ---------------------------------------------------------------------------
"""KNN + Random‑Forest ensemble with variance‑aware blending."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .distance_calculator import DistanceCalculator
from .index_backends import FaissIndex, NearestNeighborIndex, SklearnIndex
from .schemas import EngineConfig

_CFG = EngineConfig()  # pull default values only – runtime CONFIG created in __init__


@dataclass(slots=True)
class KNNModel:
    task: Literal["classification", "regression"] = "regression"
    metric: Literal["euclidean", "mahalanobis"] = "euclidean"
    k: int = _CFG.knn_k

    _X: np.ndarray | None = None
    _y: np.ndarray | None = None
    _index: NearestNeighborIndex | None = None
    _dist: DistanceCalculator | None = None

    # --------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        X = X.astype(np.float32)
        self._X, self._y = X, y.astype(np.float32)
        self._dist = DistanceCalculator(X, self.metric)
        self._index = (
            FaissIndex(X.shape[1])
            if _CFG.ann_backend == "faiss" and X.shape[0] > 5_000
            else SklearnIndex("euclidean")
        )
        self._index.fit(X)
        return self

    # internal
    def _kneighbors(self, Q):
        return self._index.kneighbors(Q.astype(np.float32), self.k)

    # --------------------------------------------------
    def predict_and_std(self, Q) -> Tuple[np.ndarray, np.ndarray]:
        idx, dist = self._kneighbors(Q)
        w = 1.0 / (dist + 1e-6)
        neigh_y = self._y[idx]
        mean = (w * neigh_y).sum(1) / w.sum(1)
        var = (w * (neigh_y - mean[:, None]) ** 2).sum(1) / w.sum(1)
        return mean, var


# --------------------------------------------------
class ModelManager:
    """Hybrid ensemble: variance‑weighted KNN + RandomForest."""

    '''def __init__(self, use_rf: bool = True):
        self.knn = KNNModel()
        self.rf: RandomForestRegressor | None = None if use_rf else None
        self.use_rf = use_rf'''

    def __init__(self, artefact_dir: str | Path, use_rf: bool = True):
        # Store the path that RetrainingManager needs
        self.artefact_dir = Path(artefact_dir)

        # Your existing logic
        self.knn = KNNModel()
        self.rf: RandomForestRegressor | None = None if use_rf else None
        self.use_rf = use_rf

    # ----------------------------------------------
    def train_model(self, X: np.ndarray, y: np.ndarray):
        self.knn.fit(X, y)
        if self.use_rf:
            self.rf = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                n_jobs=-1,
                random_state=42,
            ).fit(X, y)

    # ----------------------------------------------
    def predict(self, feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return *(mean, variance)* per sample."""
        mean_k, var_k = self.knn.predict_and_std(feats)

        if not self.use_rf or self.rf is None:
            return mean_k, var_k

        # --- RF expectations ---
        mean_rf = self.rf.predict(feats).astype(np.float32)
        # cheap variance proxy: variance across trees
        preds_per_tree = np.stack([t.predict(feats) for t in self.rf.estimators_])
        var_rf = preds_per_tree.var(0).astype(np.float32) + 1e-6

        # inverse‑variance blending
        w_k = 1.0 / var_k
        w_rf = 1.0 / var_rf
        mu = (w_k * mean_k + w_rf * mean_rf) / (w_k + w_rf)
        var = 1.0 / (w_k + w_rf)
        return mu, var


    # ----------------------------------------------
    # NEW: hot-swap artefacts after retrain
    # ----------------------------------------------
    def load_latest(self, artefact_dir: "Path") -> None:
        """
        Refresh any in-memory models that depend on the artefacts in
        *artefact_dir*.  At present the ensemble has no direct files to load,
        so this is a lightweight stub that simply logs the swap.  Extend it
        later if you keep an EVEngine instance or other artefacts in memory.
        """
        import logging
        logging.getLogger(__name__).info(
            "[ModelManager] hot-swapped artefacts from %s", artefact_dir
        )
        # Example of a real reload:
        # from prediction_engine.ev_engine import EVEngine
        # self.ev = EVEngine.from_artifacts(artefact_dir)

