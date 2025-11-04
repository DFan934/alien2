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
import warnings
from sklearn.impute import SimpleImputer
import sklearn  # for __version__ in meta

from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd

import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from collections import Counter

from prediction_engine.ev_engine import _sha1_list


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
        y_numeric: np.ndarray,
        y_categorical: Optional[pd.Series],

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

        if X.shape[0] != y_numeric.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        # --------------------------------------------------------------#
        # 0. Mean-impute NaNs  (Task-2.2)                                #
        # --------------------------------------------------------------#
        nan_mask = np.isnan(X).any(axis=1)
        ratio_nan = nan_mask.mean()
        if ratio_nan >= 0.10:
            warnings.warn(
                f"[PathClusterEngine] {ratio_nan:.1%} of rows contained NaNs; "
                "mean-imputation applied."
            )

        imputer = SimpleImputer(strategy="mean")
        X = imputer.fit_transform(X)

        # ---- Determine effective number of clusters (guard against tiny train sets)
        n_samples = int(X.shape[0])
        k_req = int(n_clusters)
        k_eff = max(1, min(k_req, n_samples))
        if k_eff < k_req:
            warnings.warn(
                f"[PathClusterEngine] Requested n_clusters={k_req} but only n_samples={n_samples}; "
                f"using k_eff={k_eff}."
            )

        # --------------------------------------------------------------#
    # 1. K-Means                                                    #
    # --------------------------------------------------------------#
        '''km = KMeans(
            n_clusters=n_clusters,
            n_init="auto",
            max_iter=300,
            random_state=random_state,
        ).fit(X.astype(np.float32))

        # ------------------------------------------------------------------
        # 2. Save centroids --------------------------------------------------
        centers = km.cluster_centers_.astype(np.float32)
        np.save(out_dir / "centers.npy", centers)

        # ------------------------------------------------------------------
        # 3. Outcome stats per cluster (mu, var) -----------------------------
        labels = km.labels_
        cluster_regime = np.full(n_clusters, "ANY", dtype="U10")  # default placeholder
        mu = np.zeros(n_clusters, dtype=np.float32)
        var = np.zeros(n_clusters, dtype=np.float32)
        var_down = np.zeros(n_clusters, dtype=np.float32)
        '''

        # ---- Fit clusters (guarded)
        X32 = X.astype(np.float32)

        if k_eff == 1:
            # Trivial single-cluster fallback
            centers = X32.mean(axis=0, keepdims=True).astype(np.float32)
            labels = np.zeros(n_samples, dtype=int)
        else:
            km = KMeans(
                n_clusters=k_eff,
                n_init="auto",
                max_iter=300,
                random_state=random_state,
            ).fit(X32)
            centers = km.cluster_centers_.astype(np.float32)
            labels = km.labels_.astype(int)

        # Persist centroids
        np.save(out_dir / "centers.npy", centers)

        # Outcome stats arrays sized by the effective k
        cluster_regime = np.full(k_eff, "ANY", dtype="U10")  # default placeholder
        mu = np.zeros(k_eff, dtype=np.float32)
        var = np.zeros(k_eff, dtype=np.float32)
        var_down = np.zeros(k_eff, dtype=np.float32)

        global_mu = float(y_numeric.mean())
        global_var = float(np.var(y_numeric))
        neg_all = y_numeric[y_numeric < 0]
        global_var_down = float(np.var(neg_all)) if neg_all.size else global_var
        outcome_probs: dict[str, dict[str, float]] = {}

        for c in range(k_eff):
            idx = labels == c
            vals = y_numeric[idx]
            tiny = vals.size < 3

            if tiny:
                mu[c] = global_mu
                var[c] = global_var
                var_down[c] = global_var_down
            else:
                mu[c] = float(vals.mean())
                var[c] = float(np.var(vals))

                neg = vals[vals < 0]
                var_down[c] = float(np.var(neg)) if neg.size >= 3 else var[c]

            if y_categorical is not None:
                cat_vals = y_categorical[idx]
                if not cat_vals.empty:
                    probs = cat_vals.value_counts(normalize=True).to_dict()
                    outcome_probs[str(c)] = probs

            if y_categorical is not None and y_categorical.dtype == object:
                # treat object‐dtype column as regime label
                regs = y_categorical[idx].str.upper()
                if not regs.empty:
                    cluster_regime[c] = Counter(regs).most_common(1)[0][0]

        np.savez_compressed(
            out_dir / "cluster_stats.npz",
            mu=mu,
            var=var,
            var_down=var_down,
            regime=cluster_regime,  # NEW
            feature_list=np.array(feature_names, dtype="U"),
        )

        if outcome_probs:
            with open(out_dir / "outcome_probabilities.json", "w", encoding="utf-8") as f:
                json.dump(outcome_probs, f, indent=2)

        # ------------------------------------------------------------------
        # 4. Kernel bandwidth (median pairwise dist of centroids) -----------
        if centers.shape[0] > 1:
            h_val: float = float(np.median(pdist(centers)))
        else:
            h_val = 1.0  # degenerate – single centroid, arbitrary h

        with open(out_dir / "kernel_bandwidth.json", "w", encoding="utf-8") as f:
            json.dump({"h": h_val}, f, indent=2)

        # ------------------------------------------------------------------
        # 5. Meta / schema hash ---------------------------------------------
        sha = hashlib.sha1("|".join(feature_names).encode()).hexdigest()[:12]
        '''meta = {
            "n_clusters": int(n_clusters),
            "features": feature_names,
            "sha": _sha1_list(feature_names),
            "sklearn_version": km.__module__.split(".")[0] + " " + km.__module__.split(".")[1],
        }'''
        meta = {
            "n_clusters": int(k_eff),  # actual used
            "k_requested": int(k_req),  # what caller asked for
            "features": feature_names,
            "sha": _sha1_list(feature_names),
            "sklearn_version": sklearn.__version__,
        }

        with open(out_dir / "meta.json", "w", encoding="utf-8") as mf:
            json.dump(meta, mf, indent=2)

        # ------------------------------------------------------------------
        # 6. NEW  – feature_schema.json  (ordering + per‑feature scale) -----
        # ------------------------------------------------------------------
        # If X has been standardised (mean‑zero / unit‑σ) upstream, store σ
        # per column; otherwise store 1.0 so the downstream check still works.
        if hasattr(imputer, "statistics_"):
            # crude std‑dev estimate post‑imputation
            scale_vec = X.std(axis=0).astype(float).tolist()
        else:
            scale_vec = [1.0] * len(feature_names)
        schema = {
            "sha": sha,
            "features": feature_names,
            "scales": scale_vec,
            }
        with open(out_dir / "feature_schema.json", "w", encoding="utf-8") as sf:
            json.dump(schema, sf, indent=2)


        return cls(centers, mu, var, var_down, h_val, feature_names, sha, cluster_regime, outcome_probs)

    @classmethod
    def load(cls, artifact_dir: Path, feature_names: List[str]) -> "PathClusterEngine":
        """Load trained artifacts; validate feature schema drift. [cite: 767]"""
        artifact_dir = Path(artifact_dir)
        centers = np.load(artifact_dir / "centers.npy")
        stats = np.load(artifact_dir / "cluster_stats.npz")
        regime_arr = stats["regime"] if "regime" in stats.files else np.full(len(stats["mu"]), "ANY", dtype="U10")

        with open(artifact_dir / "kernel_bandwidth.json", "r", encoding="utf-8") as f:
            h_val = float(json.load(f)["h"])
        with open(artifact_dir / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        # NEW: Load outcome probabilities if the file exists
        outcome_probs = {}
        probs_path = artifact_dir / "outcome_probabilities.json"
        if probs_path.exists():
            with open(probs_path, "r", encoding="utf-8") as f:
                outcome_probs = json.load(f)

        expected_sha = meta.get("sha")
        actual_sha = hashlib.sha1("|".join(feature_names).encode()).hexdigest()[:12]
        if expected_sha != actual_sha:
            raise RuntimeError(f"Feature schema mismatch!")

        return cls(centers, stats["mu"].astype(np.float32), stats["var"].astype(np.float32),
                   stats["var_down"].astype(np.float32), h_val,
                   feature_names, expected_sha, regime_arr, outcome_probs)

    # ---- core object -----------------------------------------------------

    def __init__(
        self,
        centers: np.ndarray,
        mu: np.ndarray,
        var: np.ndarray,
        var_down: np.ndarray,
        h: float,
        feature_names: List[str],
        sha: str,
        regime: np.ndarray,
        outcome_probs: dict,

    ) -> None:
        self.centers = centers
        self.mu = mu
        self.var = var
        self.var_down = var_down
        self.h = h
        self.feature_names = feature_names
        self.sha = sha
        self.cluster_regime = regime
        self.outcome_probs = outcome_probs


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

        # NEW: Helper to get outcome probabilities for a given cluster
    def probs_for(self, cluster_id: int) -> dict:
        return self.outcome_probs.get(str(cluster_id), {})