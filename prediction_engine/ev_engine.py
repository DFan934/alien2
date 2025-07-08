# ---------------------------------------------------------------------------
# prediction_engine/ev_engine.py  –  k capped by centres AND feature‐dimension
# ---------------------------------------------------------------------------
from __future__ import annotations

import hashlib

from hypothesis.extra.numpy import NDArray

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

from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from typing import Iterable, Literal, Tuple

import json
import numpy as np
from numpy.typing import NDArray

from .analogue_synth import AnalogueSynth

from .distance_calculator import DistanceCalculator
#from . import tx_cost as _tx_cost # ← alias to avoid shadowing
from .tx_cost import BasicCostModel, estimate as _legacy_estimate



__all__: Tuple[str, ...] = ("EVEngine", "EVResult")


# ---------------------------------------------------------------------------
# Public dataclass returned to ExecutionManager
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class EVResult:
    mu: float  # µ – expected return/ share
    sigma: float  # full variance (σ²)
    variance_down: float  # downside variance for Sortino / Kelly denom
    beta: NDArray[np.float32] = field(repr=False)
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

    """

    def __init__(
        self,
        *,
        centers: np.ndarray,
        mu: np.ndarray,
        var: np.ndarray,
        var_down: np.ndarray,
        h: float,
        blend_alpha: float = 0.5,  # weight on synthetic analogue (α)
        metric: Literal["euclidean", "mahalanobis", "rf_weighted"] = "euclidean",
        cov_inv: np.ndarray | None = None,
        rf_weights: np.ndarray | None = None,
        k: int | None = None,
        cost_model: BasicCostModel | None = None,

    ) -> None:
        self.centers = np.ascontiguousarray(centers, dtype=np.float32)
        self.mu = mu.astype(np.float32, copy=False)
        self.var = var.astype(np.float32, copy=False)
        self.var_down = var_down.astype(np.float32, copy=False)
        self.h = float(h)
        self.alpha = float(blend_alpha)
        self.k_max = k or 32  # fair default; will be density‑capped later
        self.metric = metric
        # Build distance helper – no external cov_inv needed (DistanceCalculator
        # computes Σ⁻¹ internally for mahalanobis).
        self._dist = DistanceCalculator(
            self.centers,
            metric=metric,
            rf_weights=rf_weights,
        )

        self._cost: BasicCostModel = cost_model or BasicCostModel()


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
        cost_model: object | None = None,
    ) -> "EVEngine":
        artefact_dir = Path(artefact_dir)
        centers = np.load(artefact_dir / "centers.npy")
        stats_path = artefact_dir / "cluster_stats.npz"
        stats = np.load(stats_path)

          # -------- sanity-check required arrays --------------------

        for key in ("mu", "var", "var_down"):
            if key not in stats.files:  # older file from previous build
                raise RuntimeError(
                    f"{stats_path} is missing '{key}'. "
                    "Delete weights directory and rebuild centroids."
                )


        #h = float(json.loads((artefact_dir / "kernel_bandwidth.json").read_text())["h"])

        # Schema guard – fail fast if feature list drifted
        meta = json.loads((artefact_dir / "meta.json").read_text())
        features_live = stats["feature_list"].tolist() if "feature_list" in stats else None
        if features_live is not None and meta["sha"] != _sha1_list(features_live):
            raise RuntimeError(
                "Feature schema drift detected – retrain PathClusterEngine before loading EVEngine."
            )

        kernel_cfg = json.loads((artefact_dir / "kernel_bandwidth.json").read_text())
        blend_alpha = kernel_cfg.get("blend_alpha", 0.5)
        h = float(kernel_cfg["h"])

        cov_inv = None
        rf_w = None
        if metric == "mahalanobis":
            cov_inv = np.load(artefact_dir / "centroid_cov_inv.npy")
        elif metric == "rf_weighted":
            rf_file = artefact_dir / "rf_feature_weights.py"
            rf_w = np.load(rf_file)
        #if not rf_path.exists():
        #        raise FileNotFoundError("RF weights file missing for rf_weighted metric")


        return cls(
            centers=centers,
            mu=stats["mu"],
            var=stats["var"],
            var_down=stats["var_down"],
            blend_alpha=blend_alpha,
            h=h,
            metric=metric,
            cov_inv=cov_inv,
            rf_weights=rf_w,
            k=k,
            cost_model=cost_model,
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def evaluate(self, x: NDArray[np.float32], adv_percentile: float | None = None) -> EVResult:  # noqa: N802
        """Return expected‑value statistics for one live feature vector."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim != 1:
            raise ValueError("x must be 1‑D vector")

        # ------------------------------------------------------------
        # 1. Find nearest neighbours (density‑aware k)
        # ------------------------------------------------------------
        k_eff = int(max(2, min(self.k_max, sqrt(len(self.centers)))))
        dist2, idx = self._dist(x, k_eff)

        # density cut‑off – remove outliers beyond 2 h
        mask = dist2 < (2 * self.h) ** 2
        if not mask.any():
            mask[:] = True  # fallback – keep all if everything out of range
        idx, dist2 = idx[mask], dist2[mask]

        # ------------------------------------------------------------
        # 2. Kernel estimate (Gaussian)
        # ------------------------------------------------------------
        w = np.exp(-0.5 * dist2 / (self.h ** 2))
        if w.sum() == 0.0:  # <-- add this
            w[:] = 1.0
        w /= w.sum()
        mu_k = float(np.dot(w, self.mu[idx]))
        var_k = float(np.dot(w, self.var[idx]))
        var_down_k = float(np.dot(w, self.var_down[idx]))

        # ------------------------------------------------------------
        # 3. Synthetic analogue via inverse‑difference weights
        # ------------------------------------------------------------
        delta_mat = self.centers[idx] - x  # (k, d)
        beta = AnalogueSynth.weights(delta_mat, -x)
        mu_syn = AnalogueSynth.synthesize(0.0, self.mu[idx], beta)
        var_syn = float(beta @ self.var[idx])  # <— one-liner
        var_down_syn = float(beta @ self.var_down[idx])

        alpha = self.alpha
        mu = alpha * mu_syn + (1 - alpha) * mu_k
        sig2 = alpha ** 2 * var_syn + (1 - alpha) ** 2 * var_k
        sig2_down = alpha ** 2 * var_down_syn + (1 - alpha) ** 2 * var_down_k

        '''# blend 50‑50
        mu = 0.5 * mu_k + 0.5 * mu_syn
        sig2 = 0.5 * var_k + 0.5 * var_syn
        sig2_down = 0.5 * var_down_k + 0.5 * var_down_syn
        '''
        '''if adv_percentile is not None:
            if adv_percentile >= 20:
                liq_fac = 1.5
            elif adv_percentile <= 5:
                liq_fac = 1.0
            else:  # linear 5 → 20 pct
                liq_fac = 1.0 + (adv_percentile - 5) * (0.5 / 15)
            sig2 *= liq_fac ** 2
            sig2_down *= liq_fac ** 2'''
        if adv_percentile is None:
            liquidity_mult = 1.0
        else:  # piece‑wise linear: 1× below 5 % ADV  → 1.5× at 20 %
            liquidity_mult = 1.0 + 0.5 * min(max((adv_percentile - 5) / 15, 0.0), 1.0)

        sig2 *= liquidity_mult
        sig2_down *= liquidity_mult

        # ------------------------------------------------------------
        # 4. Subtract transaction cost
        # ------------------------------------------------------------
        cost_ps = self._cost.estimate(adv_percentile=adv_percentile)
        mu_net = mu - cost_ps

        return EVResult(mu_net, sig2, sig2_down, beta.astype(np.float32), int(idx[0]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha1_list(items: Iterable[str]) -> str:
    h = hashlib.sha1()
    for it in items:
        h.update(it.encode())
    return h.hexdigest()
