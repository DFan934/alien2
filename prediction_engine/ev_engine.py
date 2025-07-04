# ---------------------------------------------------------------------------
# prediction_engine/ev_engine.py  –  k capped by centres AND feature‐dimension
# ---------------------------------------------------------------------------
from __future__ import annotations

import hashlib

"""Expected‑Value Engine
========================

The EVEngine turns a *feature vector* representing the *live* state of a setup
into:

* **EVResult.mu** – expected return per share
* **EVResult.sigma** – variance of return per share (full) and downside‐only
  variance (Sortino variant) for Kelly sizing
* **cluster_id** – id of the closest path cluster for dashboarding

It now supports **Euclidean** *and* **Mahalanobis** distances, variable *k*
choice based on local centroid density, automatic loading of *kernel bandwidth*
(h) and **feature‑schema guards** written by ``PathClusterEngine``.

File location:  ``prediction_engine/ev_engine.py`` – drop‑in replacement for the
old module.
"""

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Iterable, Literal, Tuple

import json
import numpy as np
from scipy.stats import kurtosis

from .distance_calculator import DistanceCalculator
#from . import tx_cost
from . import tx_cost as _tx_cost          # ← alias to avoid shadowing


# ---------------------------------------------------------------------------
# Public dataclass returned to ExecutionManager
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class EVResult:
    mu: float  # µ – expected return/ share
    sigma: float  # full variance (σ²)
    variance_down: float  # downside variance for Sortino / Kelly denom
    cluster_id: int  # closest centroid index


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class EVEngine:
    """Maps feature‑vectors to expected‑value statistics.

    Parameters
    ----------
    centers : np.ndarray
        ``(n_clusters, n_features)`` array of k‑means centroids.
    mu : np.ndarray
        Mean returns per cluster.
    var : np.ndarray
        Variance of returns per cluster.
    var_down : np.ndarray
        Downside variance per cluster.
    h : float
        Kernel bandwidth for the Gaussian kernel.
    metric : Literal["euclidean", "mahalanobis"]
        Distance metric to use.
    cov_inv : np.ndarray | None
        Pre‑computed inverse covariance for Mahalanobis.
    k : int | None
        Maximum neighbours used in kernel; will be down‑capped dynamically.
    tx_cost : tx_cost | None
        Optional slippage/commission model to be subtracted from µ.
    """

    def __init__(
        self,
        *,
        centers: np.ndarray,
        mu: np.ndarray,
        var: np.ndarray,
        var_down: np.ndarray,
        h: float,
        metric: Literal["euclidean", "mahalanobis", "rf_weighted"] = "euclidean",
        cov_inv: np.ndarray | None = None,
        k: int | None = None,
        #tx_cost: tx_cost | None = None,
        cost_model: object | None = None,

    ) -> None:
        self.centers = np.ascontiguousarray(centers, dtype=np.float32)
        self.mu = mu.astype(np.float32, copy=False)
        self.var = var.astype(np.float32, copy=False)
        self.var_down = var_down.astype(np.float32, copy=False)
        self.h = float(h)
        self.k_max = k or 32  # fair default; will be density‑capped later
        self.metric = metric
        self._dist = DistanceCalculator(self.centers, metric=metric, cov_inv=cov_inv)
        #self.tx_cost = tx_cost or tx_cost.basic()

        self._cost = cost_model if cost_model is not None else _tx_cost


        if len(self.centers) != len(self.mu):
            raise ValueError("centers and mu size mismatch")

    # ------------------------------------------------------------------
    # Factory – load from artefact directory written by PathClusterEngine
    # ------------------------------------------------------------------

    @classmethod
    def from_artifacts(
        cls,
        artefact_dir: Path | str,
        *,
        metric: Literal["euclidean", "mahalanobis", "rf_weighted"] = "euclidean",
        k: int | None = None,
        #tx_cost: tx_cost | None = None,
        cost_model: object | None = None,
    ) -> "EVEngine":
        artefact_dir = Path(artefact_dir)
        centers = np.load(artefact_dir / "centers.npy")
        #stats = np.load(artefact_dir / "cluster_stats.npz")
        stats_path = artefact_dir / "cluster_stats.npz"
        stats = np.load(stats_path)

          # -------- sanity-check required arrays --------------------

        for key in ("mu", "var", "var_down"):
            if key not in stats.files:  # older file from previous build
                raise RuntimeError(
                    f"{stats_path} is missing '{key}'. "
                    "Delete weights directory and rebuild centroids."
                )


        h = float(json.loads((artefact_dir / "kernel_bandwidth.json").read_text())["h"])

        # Schema guard – fail fast if feature list drifted
        meta = json.loads((artefact_dir / "meta.json").read_text())
        features_live = stats["feature_list"].tolist() if "feature_list" in stats else None
        if features_live is not None and meta["sha"] != _sha1_list(features_live):
            raise RuntimeError(
                "Feature schema drift detected – retrain PathClusterEngine before loading EVEngine."
            )

        cov_inv = None
        if metric == "mahalanobis":
            cov_inv = np.load(artefact_dir / "centroid_cov_inv.npy")
        elif metric == "rf_weighted":
         # just verify weights file exists – DistanceCalculator loads it lazily

            if not (artefact_dir / "weights.npy").exists():

                raise FileNotFoundError(
                                    f"metric='rf_weighted' requested but {artefact_dir / 'weights.npy'} missing")


        return cls(
            centers=centers,
            mu=stats["mu"],
            var=stats["var"],
            var_down=stats["var_down"],
            h=h,
            metric=metric,
            cov_inv=cov_inv,
            k=k,
            #tx_cost=tx_cost,
            cost_model=cost_model,
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def evaluate(self, x: np.ndarray, adv_percentile: float | None = None) -> EVResult:
        """Compute EV stats for one *live* feature vector.

        Parameters
        ----------
        x : np.ndarray
            Feature vector, shape ``(n_features,)``.
        adv_percentile : float, optional
            ADV percentile of the traded symbol (0‑100). Used for
            dynamic slippage adjustment; low‑liquidity names widen the
            assumed spread.
        """
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim != 1:
            raise ValueError("x must be 1‑D vector")

        # density‑aware k: √N but at least 2 and at most k_max
        k_eff = int(max(2, min(self.k_max, sqrt(len(self.centers)))))
        idx, dist2 = self._dist(x, k_eff)

        # Gaussian kernel weights
        w = np.exp(-0.5 * dist2 / (self.h**2))
        if w.sum() == 0.0:
            w[:] = 1  # fallback uniform if all distances large
        w /= w.sum()

        mu = float(np.dot(w, self.mu[idx]))
        sig2 = float(np.dot(w, self.var[idx]))
        sig2_down = float(np.dot(w, self.var_down[idx]))

        # subtract slippage / commission
        #cost = self.tx_cost.estimate(adv_percentile=adv_percentile)

        cost = (
            self._cost.estimate(order_size=1, ADV=1)
            if hasattr(self._cost, "estimate")
            else 0.0
        )

        mu_net = mu - cost

        return EVResult(mu_net, sig2, sig2_down, int(idx[0]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha1_list(items: Iterable[str]) -> str:
    h = hashlib.sha1()
    for it in items:
        h.update(it.encode())
    return h.hexdigest()
