# ---------------------------------------------------------------------------
# prediction_engine/path_cluster_engine.py
# ---------------------------------------------------------------------------
"""Path‑Cluster Engine
======================
Groups historical *data‑group* feature vectors into `n_clusters` using
**MiniBatchKMeans**, then persists both the cluster centroids *and* outcome
statistics `(mean, var)` required by `EVEngine`.

The module exposes two entry points:

* `PathClusterEngine.build(df, features, outcome_col, n_clusters, cfg)` –
  trains a new model and writes `centers.npy` and `cluster_stats.npz`.
* `PathClusterEngine.load(model_dir)` – returns an immutable view used by the
  online scorer.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import List, Tuple

import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans


class PathClusterEngine:
    """Trains / loads path‑space centroids & outcome stats used by EVEngine.

    The *build* classmethod trains *n_clusters* K‑Means centroids on the provided
    *feature matrix* **X** and associates each centroid with the mean (mu) and
    variance (var) of the supplied *target vector* **y**.

    Artifacts written under *out_dir* follow a strict schema so that
    EVEngine.from_artifacts() can consume them without ambiguity:

    * centers.npy                – (n_clusters, n_features) float32 centroids
    * cluster_stats.npz          – keys "mu", "var"  float32 arrays length n_clusters
    * kernel_bandwidth.json      – {"h": <float>}  median pairwise distance of centroids
    * meta.json                  – {"n_clusters": int, "features": List[str],
                                     "sha": str (sha1 of feature list),
                                     "sklearn_version": str}
    """

    # ---- public helpers --------------------------------------------------

    @classmethod
    def build(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        n_clusters: int,
        out_dir: Path,
        random_state: int | None = 17,
    ) -> "PathClusterEngine":
        """Train clusters + stats and persist artifacts.

        Parameters
        ----------
        X : (n_samples, n_features)
            Feature matrix **after PCA** / scaling.
        y : (n_samples,)
            Realised outcomes (future log‑returns, EV proxy, etc.).  *Must* align
            1‑to‑1 with **X** rows.
        feature_names : list[str]
            Ordered list of feature column names so that subsequent inference
            stages can cross‑check for schema drift.
        n_clusters : int
            Number of K‑Means centroids.
        out_dir : pathlib.Path
            Destination folder for artifacts. Will be created if missing.
        random_state : int | None, default=17
            Reproducible RNG seed.
        """
        out_dir.mkdir(parents=True, exist_ok=True)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        km = KMeans(
            n_clusters=n_clusters,
            n_init="auto",
            max_iter=300,
            random_state=random_state,
        ).fit(X.astype(np.float32))

        # ------------------------------------------------------------------
        # 1. Save centroids --------------------------------------------------
        centers = km.cluster_centers_.astype(np.float32)
        np.save(out_dir / "centers.npy", centers)

        # ------------------------------------------------------------------
        # 2. Outcome stats per cluster (mu, var) -----------------------------
        labels = km.labels_
        mu = np.zeros(n_clusters, dtype=np.float32)
        var = np.zeros(n_clusters, dtype=np.float32)
        global_mu = float(y.mean())
        global_var = float(np.var(y))

        for c in range(n_clusters):
            idx = labels == c
            cluster_vals = y[idx]
            if cluster_vals.size < 3:
                # Not enough data – fall back to global priors per critique a/b.
                mu[c] = global_mu
                var[c] = global_var
            else:
                mu[c] = float(cluster_vals.mean())
                var[c] = float(np.var(cluster_vals))


        # downside variance (only negative outcomes)
        var_down = np.zeros(n_clusters, dtype=np.float32)
        for c in range(n_clusters):
            neg = y[labels == c]
            neg = neg[neg < 0]
            var_down[c] = float(np.var(neg)) if neg.size >= 3 else global_var

        np.savez_compressed(out_dir / "cluster_stats.npz", mu=mu, var=var, var_down=var_down)

        # ------------------------------------------------------------------
        # 3. Kernel bandwidth (median pairwise dist of centroids) -----------
        if centers.shape[0] > 1:
            h_val: float = float(np.median(pdist(centers)))
        else:
            h_val = 1.0  # degenerate – single centroid, arbitrary h

        with open(out_dir / "kernel_bandwidth.json", "w", encoding="utf-8") as f:
            json.dump({"h": h_val}, f, indent=2)

        # ------------------------------------------------------------------
        # 4. Meta / schema hash ---------------------------------------------
        sha = hashlib.sha1("|".join(feature_names).encode()).hexdigest()[:12]
        meta = {
            "n_clusters": int(n_clusters),
            "features": feature_names,
            "sha": sha,
            "sklearn_version": km.__module__.split(".")[0] + " " + km.__module__.split(".")[1],
        }
        with open(out_dir / "meta.json", "w", encoding="utf-8") as mf:
            json.dump(meta, mf, indent=2)

        return cls(centers, mu, var, h_val, feature_names, sha)

    @classmethod
    def load(cls, artifact_dir: Path, feature_names: List[str]) -> "PathClusterEngine":
        """Load trained artifacts; validate feature schema drift."""
        centers = np.load(artifact_dir / "centers.npy")
        stats = np.load(artifact_dir / "cluster_stats.npz")
        with open(artifact_dir / "kernel_bandwidth.json", "r", encoding="utf-8") as f:
            h_val = float(json.load(f)["h"])
        with open(artifact_dir / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        # --- schema guard --------------------------------------------------
        expected_sha = meta.get("sha")
        actual_sha = hashlib.sha1("|".join(feature_names).encode()).hexdigest()[:12]
        if expected_sha != actual_sha:
            raise RuntimeError(
                "Feature schema mismatch between training (sha=%s) and runtime (sha=%s)"
                % (expected_sha, actual_sha)
            )

        mu = stats["mu"].astype(np.float32)
        var = stats["var"].astype(np.float32)
        return cls(centers, mu, var, h_val, feature_names, expected_sha)

    # ---- core object -----------------------------------------------------

    def __init__(
        self,
        centers: np.ndarray,
        mu: np.ndarray,
        var: np.ndarray,
        h: float,
        feature_names: List[str],
        sha: str,
    ) -> None:
        self.centers = centers
        self.mu = mu
        self.var = var
        self.h = h
        self.feature_names = feature_names
        self.sha = sha

    # ---- helpers ---------------------------------------------------------

    def closest_centroid(self, x: np.ndarray) -> Tuple[int, float]:
        """Return (cluster_id, euclidean_distance)."""
        dists = norm(self.centers - x, axis=1)
        cid = int(np.argmin(dists))
        return cid, float(dists[cid])

    # ---------------------------------------------------------------------
    # Convenience for EVEngine: returns mu, var of closest cluster ----------
    # ---------------------------------------------------------------------

    def stats_for(self, x: np.ndarray) -> Tuple[float, float]:
        cid, _ = self.closest_centroid(x)
        return float(self.mu[cid]), float(self.var[cid])
