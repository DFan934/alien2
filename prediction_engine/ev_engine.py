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
from pathlib import Path
from typing import Iterable, Literal, Tuple, Dict
import joblib
from prediction_engine.position_sizer import KellySizer        # NEW
from .drift_monitor import get_monitor, DriftStatus        # NEW

import json
import numpy as np
from numpy.typing import NDArray
from .distance_calculator import DistanceCalculator
from .weight_optimization import WeightOptimizer
from .market_regime import MarketRegime
from .analogue_synth import AnalogueSynth

from .distance_calculator import DistanceCalculator
from .tx_cost import BasicCostModel
from .market_regime import MarketRegime # NEW
from .weight_optimization import CurveParams, WeightOptimizer  # NEW



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
    residual: float

    cluster_id: int  # closest centroid index
    outcome_probs: Dict[str, float] = field(default_factory=dict)   # flat dict
    #regime_curves: Dict[str, CurveParams] | None = None,
    position_size: float = 0.0
    drift_ticket: int = -1            # NEW – link to DriftMonitor
    # NEW – Kelly size





# ------------------------------------------------------------------
# internal async retrain trigger
# ------------------------------------------------------------------
def _kickoff_retrain() -> None:
    """Call PathClusterEngine.build() in a background thread."""
    import logging, time
    from prediction_engine.path_cluster_engine import PathClusterEngine
    log = logging.getLogger(__name__)

    # crude debounce: only allow one retrain per hour
    tag_file = Path("prediction_engine/artifacts/last_retrain.txt")
    if tag_file.exists() and time.time() - tag_file.stat().st_mtime < 3600:
        return

    log.warning("[Drift] RETRAIN_SIGNAL received – rebuilding centroids …")
    # (*real code would pull latest parquet, rebuild, then atomically swap artefacts*)
    try:
        #  --- demo stub: reload existing artefacts as “new” ---
        artefacts = Path("prediction_engine/artifacts")
        # NOTE: insert actual ETL + PathClusterEngine.build() call here
        PathClusterEngine.build(
            X=np.load(artefacts / "centers.npy"),        # placeholder
            y_numeric=np.load(artefacts / "cluster_stats.npz")["mu"],
            y_categorical=None,
            feature_names=["f1","f2"],
            n_clusters=8,
            out_dir=artefacts
        )
        tag_file.touch()
        log.warning("[Drift] retrain completed.")
    except Exception:           # noqa: BLE001
        log.exception("Background retrain failed")



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
        outcome_probs: Dict[str, Dict[str, float]] | None = None,
        regime_curves: Dict[str, CurveParams] | None = None,
        blend_alpha: float = 0.5,  # weight on synthetic analogue (α)
        lambda_reg: float = 0.05,  # NEW: ridge shrinkage weight (0 = none)
        residual_threshold: float = 0.001,  # NEW: max acceptable synth residual
        metric: Literal["euclidean", "mahalanobis", "rf_weighted"] = "euclidean",

            cov_inv: np.ndarray | None = None,
        rf_weights: np.ndarray | None = None,
        k: int | None = None,
        cost_model: BasicCostModel | None = None,
        cluster_regime: np.ndarray | None = None,

    ) -> None:
        self.centers = np.ascontiguousarray(centers, dtype=np.float32)
        self.mu = mu.astype(np.float32, copy=False)
        self.var = var.astype(np.float32, copy=False)
        self.var_down = var_down.astype(np.float32, copy=False)
        self.h = float(h)
        self.alpha = float(blend_alpha)
        self.lambda_reg = float(lambda_reg)  # NEW
        # NEW: threshold for acceptable synth residual
        self.residual_threshold = float(residual_threshold)

        self.k_max = k or 32  # fair default; will be density‑capped later
        self.metric = metric
        '''# Build distance helper – no external cov_inv needed (DistanceCalculator
        # computes Σ⁻¹ internally for mahalanobis).
        self._dist = DistanceCalculator(
            self.centers,
            metric=metric,
            rf_weights=rf_weights,
        )'''
        # NEW: Store regime-specific curves and outcome probabilities
        self.regime_curves = regime_curves or {}
        self.outcome_probs = outcome_probs or {}

        # Build distance helper – inject recency weights per current regime
        # Pull the active regime (enum) and its CurveParams
        #regime_cfg = self.regime_curves.get(MarketRegime[regime.upper()].name.lower())
        # Compute a recency‐weight vector matching feature‐dimensionality
        #if regime_cfg:
        #    n_feat = self.centers.shape[1]
        #    recency_w = WeightOptimizer._weights(n_feat, regime_cfg)
        #else:
        #    recency_w = None

        # Build distance helper – core k‑NN search (recency comes later)
        self._dist = DistanceCalculator(
            ref = self.centers,
            metric = metric,
            rf_weights = rf_weights,
        )

        self._cost: BasicCostModel = cost_model or BasicCostModel()

        # Kelly-sizer helper (singleton inside engine)
        self._sizer = KellySizer()



        #NEW: Loading the meta boost model
        model_path = Path("prediction_engine/artifacts/gbm_meta.pkl")
        self._meta_model = joblib.load(model_path) if model_path.exists() else None

        #NEW: Cluster regime
        self.cluster_regime = cluster_regime if cluster_regime is not None else np.full(len(mu), "ANY", dtype="U10")

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
        stats = np.load(stats_path, allow_pickle=True)
        regime_arr = stats["regime"] if "regime" in stats.files else np.full(len(stats["mu"]), "ANY", dtype="U10")

        # ─── load all per‐regime curves ────────────────────
        from prediction_engine.weight_optimization import load_regime_curves
        weight_root = Path("artifacts") / "weights"
        curves = {}
        if weight_root.exists():
            curves = load_regime_curves(weight_root)


        # NEW: Load all available regime-specific weighting curves
        regime_curves: dict[str, CurveParams] = {}
        for regime_dir in artefact_dir.glob("regime=*"):
            regime_name = regime_dir.name.split("=")[-1]
            params_file = regime_dir / "curve_params.json"
            if params_file.exists():
                params_data = json.loads(params_file.read_text())["params"]
                regime_curves[regime_name] = CurveParams(**params_data)

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
        features_live = stats["feature_list"].tolist()

        print("[DEBUG] meta['sha']:", meta["sha"])
        print("[DEBUG] current sha:", _sha1_list(features_live))
        print("[DEBUG] meta['features']:", meta.get("features"))
        print("[DEBUG] current features:", features_live)

        if meta["sha"] != _sha1_list(features_live):            raise RuntimeError(
                "Feature schema drift detected – retrain PathClusterEngine before loading EVEngine."
            )

        kernel_cfg = json.loads((artefact_dir / "kernel_bandwidth.json").read_text())
        blend_alpha = kernel_cfg.get("blend_alpha", 0.5)
        h = float(kernel_cfg["h"])

        # NEW: Load outcome probabilities
        probs_path = artefact_dir / "outcome_probabilities.json"
        outcome_probs = json.loads(probs_path.read_text()) if probs_path.exists() else {}

        # NEW: Load all available regime-specific weighting curves
        '''regime_curves = {}
        for regime_dir in artefact_dir.glob("regime=*"):
            regime_name = regime_dir.name.split("=")[-1]
            params_file = regime_dir / "curve_params.json"
            if params_file.exists():
                params_data = json.loads(params_file.read_text())["params"]
                regime_curves[regime_name] = CurveParams(**params_data)'''

        # ─── Pull in ALL regime curves generated by nightly_calibrate.py ───
        #from prediction_engine.weight_optimization import load_all_regime_curves
        #regime_curves = load_all_regime_curves(artefact_dir / "weights")

        #from prediction_engine.weight_optimization import load_regime_curves
        '''weight_root = Path("artifacts") / "weights"
        if weight_root.exists():
            try:
                extra = load_regime_curves(weight_root)
                regime_curves.update(extra)
            except Exception:
                # ignore parse errors
        
                pass'''

        # ────────────────────────────────────────────────────────────
        '''#Merge in the nightly_calibrate outputs from artifacts/weights
        from pathlib import Path as _P
        try:
            from prediction_engine.weight_optimization import load_regime_curves
        except ImportError:
            load_regime_curves = None
        weight_root = _P("artifacts") / "weights"
        if load_regime_curves and weight_root.exists():
            # load all regime curves from nightly calibrator
            try:
                extra = load_regime_curves(weight_root)
                regime_curves.update(extra)
            except Exception:
                # if anything goes wrong parsing these files, just skip
                pass'''

        # ─── Merge in nightly_calibrate’s per‑regime curves ───
        #from pathlib import Path
        from prediction_engine.weight_optimization import load_regime_curves
        weight_root = Path("artifacts") / "weights"
        if weight_root.exists():
            try:
                regime_curves.update(load_regime_curves(weight_root))
            except Exception:
                # ignore parsing errors
                pass

        cov_inv = None
        rf_w = None
        if metric == "mahalanobis":
            cov_inv = np.load(artefact_dir / "centroid_cov_inv.npy")
        elif metric == "rf_weighted":
            rf_file = artefact_dir / "rf_feature_weights.npy"
            rf_w = np.load(rf_file, allow_pickle=False)
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
            outcome_probs=outcome_probs,
            #regime_curves=regime_curves,
            regime_curves={**regime_curves, **curves},
            cost_model=cost_model,
            cluster_regime=regime_arr,
        )

    def _get_recency_weights(self, regime: MarketRegime, n_points: int) -> np.ndarray:
        """NEW: Generate recency weights based on the current market regime."""
        #curve = self.regime_curves.get(regime.name.lower())
        # Allow either lower- or upper-case keys ("trend" / "TREND")

        # ─── If no regime filtering, use uniform recency weights ───
        if regime is None:
            # uniform weights when skipping regime curves
            return np.ones(n_points, dtype=np.float32)

        curve = (
             self.regime_curves.get(regime.name.lower())
            or self.regime_curves.get("all")
                )

        #curve = (self.regime_curves.get(regime.name.lower())
        #        or self.regime_curves.get(regime.name.upper())
        #        or self.regime_curves.get(regime.name.capitalize()))
        if not curve:
            return np.ones(n_points)  # Fallback to uniform weights

        return WeightOptimizer._weights(n_points, curve)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def evaluate(self, x: NDArray[np.float32],
                 adv_percentile: float | None = None,
                 *,
                 half_spread: float | None = None,
                 regime: MarketRegime | None = None) -> EVResult:  # noqa: N802
        """Return expected‑value statistics for one live feature vector."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim != 1:
            raise ValueError("x must be 1‑D vector")

        # ------------------------------------------------------------
        # 1. Find nearest neighbours (density‑aware k)
        # ------------------------------------------------------------
        k_eff = int(max(2, min(self.k_max, (len(self.centers))**0.5)))
        # Corrected version
        idx, dist2 = self._dist(x, k_eff)
        # density cut‑off – remove outliers beyond 2 h
        mask = dist2 < (2 * self.h) ** 2
        if not mask.any():
            mask[:] = True
        idx, dist2 = idx[mask], dist2[mask]

        # --- regime filter ---------------------------------------------------
        if regime is not None:
            reg_mask = self.cluster_regime[idx] == regime.name.upper()
            if reg_mask.any():  # keep matches when possible
                idx, dist2 = idx[reg_mask], dist2[reg_mask]
            else:  # ▲ NEW soft-fallback
                import logging
                logging.getLogger(__name__).warning(
                    "[EVEngine] regime=%s yielded zero matches; "
                    "falling back to all clusters",
                    regime.name,
                )
                # fall back to full idx/dist2 already computed above
        # ---------------------------------------------------------------------

        # (distance threshold still applied earlier; no need to recompute)

        # ---------------------------------------------------------------------

        if idx.size == 0:  # << add
            import logging
            logging.getLogger(__name__).warning(
                "[EVEngine] regime=%s had zero matches; falling back to all clusters",
                regime.name
            )
            idx, dist2 = self._dist(x, k_eff)  # recompute on *all* clusters

        # ------------------------------------------------------------
        # 2. Kernel estimate (Gaussian) with NEW regime-based recency weighting
        # ------------------------------------------------------------
        recency_w = self._get_recency_weights(regime, len(dist2))
        kernel_w = np.exp(-0.5 * dist2 / (self.h ** 2))
        w = kernel_w * recency_w # Combine kernel with recency

        if w.sum() < 1e-8:  # <-- add this
            w[:] = 1.0
        w /= w.sum()
        mu_k = float(np.dot(w, self.mu[idx]))
        var_k = float(np.dot(w, self.var[idx]))
        var_down_k = float(np.dot(w, self.var_down[idx]))

        # ------------------------------------------------------------
        # 3. Synthetic analogue via inverse‑difference weights
        # ------------------------------------------------------------
        delta_mat = self.centers[idx] - x  # (k, d)
        # compute inverse-difference weights plus fit residual
        beta, residual = AnalogueSynth.weights(
            delta_mat,
            -x,
            var_nn=self.var[idx],
            lambda_ridge=self.lambda_reg,
        )

        import logging
        # If fit residual is too high, fallback to pure kernel EV
        if residual > self.residual_threshold:
            logging.getLogger(__name__).warning(
                "[EVEngine] high synth residual %.3g > %.3g; using kernel-only EV",
                residual, self.residual_threshold
            )
            # degenerate β: pick nearest neighbour only
            beta = np.zeros_like(beta)
            beta[0] = 1.0
            mu_syn = mu_k
            var_syn = var_k
            var_down_syn = var_down_k
        else:
            # standard synthetic‐analogue path
            mu_syn = float(beta @ self.mu[idx])
            mu_syn = (1.0 - self.lambda_reg) * mu_syn + self.lambda_reg * float(self.mu.mean())

            # ensemble meta-model override
            if hasattr(self, "_meta_model") and self._meta_model is not None:
                feat_meta = np.array([[mu_syn, mu_k, var_syn := float(beta @ self.var[idx]), var_k]])
                mu_meta = float(self._meta_model.predict(feat_meta))
                mu_syn = 0.3 * mu_meta + 0.7 * mu_syn

            var_syn = float(beta @ self.var[idx])
            var_down_syn = float(beta @ self.var_down[idx])


        # ▶ NEW: variance‑weighted β – favour low‑risk neighbours ◀
        '''var_arr = np.maximum(self.var[idx], 1e-12)
        beta /= var_arr  # down‑weight high‑variance clusters
        beta = np.maximum(beta, 0.0)
        beta /= beta.sum()  # renormalise to Σβ = 1'''

        '''mu_syn = float(beta @ self.mu[idx])
        mu_syn = (1.0 - self.lambda_reg) * mu_syn + self.lambda_reg * float(self.mu.mean()) #NEW

        # ---- ensemble meta-model override ----------------------------------
        if hasattr(self, "_meta_model") and self._meta_model is not None:
            feat_meta = np.array([[mu_syn, mu_k, var_syn := float(beta @ self.var[idx]), var_k]])
            mu_meta = float(self._meta_model.predict(feat_meta))
            mu_syn = 0.3 * mu_meta + 0.7 * mu_syn  # gamma = 0.3 for now


        var_syn = float(beta @ self.var[idx])  # <— one-liner
        var_down_syn = float(beta @ self.var_down[idx])'''

        alpha = self.alpha
        mu = alpha * mu_syn + (1 - alpha) * mu_k
        sig2 = alpha ** 2 * var_syn + (1 - alpha) ** 2 * var_k
        sig2_down = alpha ** 2 * var_down_syn + (1 - alpha) ** 2 * var_down_k


        #if adv_percentile is None:
        #    liquidity_mult = 1.0
        #else:  # piece‑wise linear: 1× below 5 % ADV  → 1.5× at 20 %
        #    liquidity_mult = 1.0 + 0.5 * min(max((adv_percentile - 5) / 15, 0.0), 1.0)

        # Always use flat volatility during debugging
        liquidity_mult = 1.0

        sig2 *= liquidity_mult
        sig2_down *= liquidity_mult

        # ------------------------------------------------------------
        # 4. Subtract transaction cost
        # ------------------------------------------------------------
        cost_ps = self._cost.estimate(half_spread=half_spread, adv_percentile=adv_percentile)
        mu_net = mu - cost_ps


        #5. NEW: Calculate blended probabilistic outcomes
        blended_probs = {}
        all_labels = set(label for probs in self.outcome_probs.values() for label in probs.keys())

        # Kernel-weighted probabilities
        probs_k = {label: 0.0 for label in all_labels}
        for i, cluster_idx in enumerate(idx):
            for label, prob in self.outcome_probs.get(str(cluster_idx), {}).items():
                probs_k[label] += w[i] * prob

        # Synthetic analogue-weighted probabilities
        probs_syn = {label: 0.0 for label in all_labels}
        for i, cluster_idx in enumerate(idx):
            for label, prob in self.outcome_probs.get(str(cluster_idx), {}).items():
                probs_syn[label] += beta[i] * prob

        # Final blend
        for label in all_labels:
            blended_probs[label] = (
                self.alpha * probs_syn[label] +
                (1.0 - self.alpha) * probs_k[label]
            )

        # --- ensure probabilities sum to 1 --------------------------
        tot = sum(blended_probs.values())
        if tot > 0:
            for k in blended_probs:
                 blended_probs[k] /= tot


        # ------------------------------------------------------------
        # 6. Kelly position size (NEW)
        # ------------------------------------------------------------
        pos_size = self._sizer.size(mu_net, sig2_down ** 0.5, adv_percentile)

        # ------------------------------------------------------------
        # 7. Drift-monitor registration  (prob ≈ P(μ>0) assuming normal)
        # ------------------------------------------------------------
        from math import erf, sqrt as _sqrt

        p_up = 0.5 * (1.0 + erf(mu / (_sqrt(2.0 * sig2) + 1e-9)))
        dm = get_monitor()
        ticket_id = dm.log_pred(p_up)  # OMS must later call
        # dm.log_outcome(ticket_id, realised_pnl > 0)

        if dm.status()[0] is DriftStatus.RETRAIN_SIGNAL:
            from threading import Thread
            Thread(target=_kickoff_retrain, daemon=True).start()

        return EVResult(
            mu_net,  # mu
            sig2,  # sigma
            sig2_down,  # variance_down
            beta,  # beta array
            residual,  # residual
            int(idx[0]),  # cluster_id
            blended_probs,  # outcome_probs
            pos_size,  # position_size
            ticket_id  # drift_ticket
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha1_list(items: Iterable[str]) -> str:

    feat_str = "|".join(items)
    full = hashlib.sha1(feat_str.encode()).hexdigest()
    return full[:12]
