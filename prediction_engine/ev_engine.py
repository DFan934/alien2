# ---------------------------------------------------------------------------
# prediction_engine/ev_engine.py  –  k capped by centres AND feature‐dimension
# ---------------------------------------------------------------------------
from __future__ import annotations

import hashlib

# remove: from hypothesis.extra.numpy import NDArray
from numpy.typing import NDArray
from sklearn.isotonic import IsotonicRegression

from .calibration import calibrate_isotonic

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
# ⬆ after the last std-lib import near the top of the file
import logging                         # ← ADD
log = logging.getLogger(__name__)      # ← ADD

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Tuple, Dict, Any
import joblib
from prediction_engine.position_sizer import KellySizer        # NEW
from .drift_monitor import get_monitor, DriftStatus        # NEW
from typing import Iterable, Literal, Tuple, Dict, Any
from functools import lru_cache
import json
import numpy as np
from numpy.typing import NDArray
from .distance_calculator import DistanceCalculator
from .weight_optimization import WeightOptimizer
from .market_regime import MarketRegime
from .analogue_synth import AnalogueSynth
from .calibration import map_mu_to_prob


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

    # NEW ↓
    mu_raw: float = 0.0
    p_up: float = 0.5
    p_source: str = "kernel"  # <-- ADD THIS LINE





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
        #residual_threshold: float = 0.001,  # NEW: max acceptable synth residual
        #tau_dist: float = float("inf"),
        tau_dist: float | None = None,
        max_cached: int = 1024,
        calibrator: Any | None=None,
        residual_threshold: float = 0.75,
        metric: Literal["euclidean", "mahalanobis", "rf_weighted"] = "euclidean",

            cov_inv: np.ndarray | None = None,
        rf_weights: np.ndarray | None = None,
        k: int | None = None,
        cost_model: BasicCostModel | None = None,
        cluster_regime: np.ndarray | None = None,
        _scale_vec: np.ndarray | None = None,  # internal
        _n_feat_schema: int | None = None,  # internal

    ) -> None:
        self.centers = np.ascontiguousarray(centers, dtype=np.float32)
        self.mu = mu.astype(np.float32, copy=False)
        self.var = var.astype(np.float32, copy=False)
        self.var_down = var_down.astype(np.float32, copy=False)
        self.h = float(h)
        self.alpha = float(blend_alpha)
        self.lambda_reg = float(lambda_reg)  # NEW
        # NEW: threshold for acceptable synth residual
        #self.tau_dist = float(tau_dist)
        # ─── sensible fallback for τ² if caller left it None ─────────────
        if tau_dist is None:
            self.tau_dist = (2.0 * self.h) ** 2     # ≈ kernel cut‑off radius
        else:
            self.tau_dist = float(tau_dist)

        self._synth_cache = lru_cache(max_cached)(self._synth_ev)
        self._scale_vec: np.ndarray | None = None
        self._n_feat_schema: int | None = None
        self.residual_threshold = float(residual_threshold)
        self._calibrator = calibrator
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

        self._cost = cost_model

        self._dist = DistanceCalculator(
            ref = self.centers,
            metric = metric,
            rf_weights = rf_weights,
        )

        #self._cost: BasicCostModel = cost_model or BasicCostModel()

        # --- COST MODEL SANITY (NEW) ---
        # If no cost model is provided, do not silently construct one.
        # Costs will be treated as zero in evaluate().
        #if getattr(self, "_cost", None) is not None:  # <-- This now works correctly
        ## --- COST MODEL SANITY (NEW) ---
        # If no cost model is provided, do not silently construct one.
        # Costs will be treated as zero in evaluate().
        if getattr(self, "_cost", None) is not None:
            # Document expected unit explicitly: USD per share (not fraction).
            self._cost_unit = "usd_per_share"
        else:
            self._cost_unit = "none"


        # Kelly-sizer helper (singleton inside engine)
        self._sizer = KellySizer()

        # --- feature‑contract info ---------------------------------
        self._scale_vec = (
            np.ascontiguousarray(_scale_vec, dtype=np.float32)
            if _scale_vec is not None else None
        )
        self._n_feat_schema = int(_n_feat_schema) if _n_feat_schema else None



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

    # ------------------------------------------------------------------
    # helper – compute synthetic μ/σ² once and cache via lru_cache
    # ------------------------------------------------------------------
    def _synth_ev(
            self,
            x_tup: Tuple[float, ...],
            idx_tup: Tuple[int, ...],
            mu_k: float,
            var_k: float,
            var_down_k: float,
            ) -> tuple[float, float, float, np.ndarray, float]:
            """Heavy analogue‑synthesis math (ex‑`evaluate` body)."""

            x = np.array(x_tup, dtype=np.float32)  # Convert tuple back to array

            idx = np.fromiter(idx_tup, dtype=np.intp)

            delta_mat = self.centers[idx] - x  # (k,d)
            beta, residual = AnalogueSynth.weights(
                delta_mat,
                -x,
                var_nn = self.var[idx],
                lambda_ridge = self.lambda_reg,
            )


            if residual > self.residual_threshold:  # kernel fallback
                beta[:] = 0.0
                beta[0] = 1.0
                mu_syn, var_syn, var_down_syn = mu_k, var_k, var_down_k
            else:
                mu_syn = float(beta @ self.mu[idx])
                var_syn = float(beta @ self.var[idx])
                var_down_syn = float(beta @ self.var_down[idx])

            print(f"[EV] synth residual={residual:.5g}  α={self.alpha:.2f}")
            print(f"[EV] β (first 5)={np.round(beta[:5], 4).tolist()}")
            print(f"[EV] synth μ={mu_syn:.6f}  σ²={var_syn:.6f}")

            return mu_syn, var_syn, var_down_syn, beta.astype(np.float32, copy=False), residual

    @classmethod
    def from_artifacts(
        cls,
        artefact_dir: Path | str,
        *,
        metric: Literal["euclidean", "mahalanobis", "rf_weighted"] = "euclidean",
        k: int | None = None,
        cost_model: object | None = None,
        calibrator: Any | None=None,
        residual_threshold: float = 0.75,
    ) -> "EVEngine":
        artefact_dir = Path(artefact_dir)
        centers = np.load(artefact_dir / "centers.npy")

        # ── Compute τ² as the 75‑th‑percentile of inter‑centroid distances ──
        pair_d2 = ((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1).ravel()
        #tau_sq_p75 = float(np.percentile(pair_d2, 75))
        tau_sq_p75 = float(np.percentile(pair_d2, 85))


        # ↓ ADD THIS LINE – keep only the closest ≈20 % analogues
        #tau_sq_p75 *= 0.40  # tighten residual gate

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

        #kernel_cfg = json.loads((artefact_dir / "kernel_bandwidth.json").read_text())


        # ------------------------------------------------------------------
        # NEW ▼ load feature_schema.json and validate                       |
        # ------------------------------------------------------------------
        schema_file = artefact_dir / "feature_schema.json"
        if not schema_file.exists():
            raise RuntimeError("feature_schema.json missing – rebuild centroids.")
        schema = json.loads(schema_file.read_text())
        expected_sha = schema["sha"]
        live_sha = _sha1_list(stats["feature_list"].tolist())
        if expected_sha != live_sha:
            raise RuntimeError(
                f"Feature schema drift!  artefact sha={expected_sha}, "
                 f"live sha={live_sha}"
                )

        scale_vec = np.asarray(schema.get("scales", []), dtype=np.float32)
        n_feat_schema = len(schema["features"])
        feature_names = list(schema["features"])
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


        #return cls(
        engine = cls(
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
            calibrator=calibrator,
            #residual_threshold=residual_threshold,
            residual_threshold=tau_sq_p75,

            _scale_vec=scale_vec,
            _n_feat_schema=n_feat_schema,
            cluster_regime=regime_arr,
        )

        # persist names for execution manager checks
        engine._feature_names = feature_names  # type: ignore[attr-defined]
        return engine

    # expose feature names loaded from schema for downstream enforcement
    def feature_names(self) -> list[str] | None:
        try:
            # recover from cached state: schema length implies presence
            schema_path = Path(self.__dict__.get("_EVEngine__artefact_dir", ""))
        except Exception:
            schema_path = None
        # store at construction time if needed (simpler):

        return getattr(self, "_feature_names", None)

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

        # ─── DEBUG: vector fingerprint & schema length ───────────────────────
        vec_sha = hashlib.sha1(x.tobytes()).hexdigest()[:10]
        print(f"[EV] start  vec_len={x.size}  sha={vec_sha}  expected_len={self._n_feat_schema}")
        # ─────────────────────────────────────────────────────────────────────

        # ---- feature‑length / scale guard --------------------------
        if (self._n_feat_schema is not None) and (x.size != self._n_feat_schema):
            raise ValueError(
                f"Feature length {x.size} ≠ schema length {self._n_feat_schema}"
                )
        if self._scale_vec is not None:
            x = x * self._scale_vec  # element‑wise rescale

        if x.ndim != 1:
            raise ValueError("x must be 1‑D vector")

        # ------------------------------------------------------------
        # 1. Find nearest neighbours (density‑aware k)
        # ------------------------------------------------------------
        #k_eff = int(max(2, min(self.k_max, (len(self.centers))**0.5)))
        # Corrected version
        #idx, dist2 = self._dist(x, k_eff)

        # start with a generous k_max, but choose effective k by kernel mass
        k_cap = int(self.k_max)
        idx_all, dist2_all = self._dist(x, k_cap)

        ker = np.exp(-0.5 * dist2_all / (self.h ** 2))
        order = np.argsort(dist2_all)
        ker_sorted = ker[order]
        cum = np.cumsum(ker_sorted) / max(1e-12, ker_sorted.sum())
        k_eff = int(np.searchsorted(cum, 0.80) + 1)  # 80% mass
        k_eff = max(8, min(k_eff, k_cap))

        idx, dist2 = idx_all[order[:k_eff]], dist2_all[order[:k_eff]]

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
                log.warning(
                    "[EVEngine] regime=%s yielded zero matches; "
                    "falling back to all clusters",
                    regime.name,
                    )
                # fall back to full idx/dist2 already computed above
        # ---------------------------------------------------------------------

        # (distance threshold still applied earlier; no need to recompute)

        # ---------------------------------------------------------------------

        regime_name = getattr(regime, "name", "ANY")

        if idx.size == 0:  # << add
            log.warning(
                "[EVEngine] regime=%s had zero matches; falling back to all clusters",
                regime.name
            )
            idx, dist2 = self._dist(x, k_eff)  # recompute on *all* clusters

        # ------------------------------------------------------------
        # 2. Kernel estimate (Gaussian) with NEW regime-based recency weighting
        # ------------------------------------------------------------
        '''recency_w = self._get_recency_weights(regime, len(dist2))
        kernel_w = np.exp(-0.5 * dist2 / (self.h ** 2))
        w = kernel_w * recency_w # Combine kernel with recency

        if w.sum() < 1e-8:  # <-- add this
            w[:] = 1.0
        w /= w.sum()'''

        # --- Base kernel (you can use 'ker' from the adaptive-k block above) ---
        #kernel_w = np.asarray(ker, dtype=float)
        '''kernel_w = ker_sorted[:k_eff].astype(float)

        # --- Recency weights (your helper, if any) ---
        recency_w = self._recency(idx) if hasattr(self, "_recency") and self._recency is not None else 1.0
        if np.isscalar(recency_w):
            recency_w = np.ones_like(kernel_w) * float(recency_w)

        # --- Outlier down-weighting in standardized feature space ---
        # Requires: self._scale_vec (feature stds), self.centers (neighbor feature matrix)
        zcap = getattr(self, "outlier_z_cap", 3.0)
        if getattr(self, "_scale_vec", None) is not None and self._scale_vec.size == self.centers.shape[1]:
            # standardized deltas for chosen neighbors
            diff = (self.centers[idx] - x) / (self._scale_vec + 1e-12)
            zmax = np.max(np.abs(diff), axis=1)
            outlier_mult = np.where(zmax > zcap, np.exp(-0.5 * (zmax - zcap)), 1.0)
        else:
            outlier_mult = 1.0

        # --- Final neighbor weights ---
        #w = kernel_w * recency_w * outlier_mult
        #w_sum = float(w.sum())

        # --- Recency weights (per-neighbor), robust to shape mismatch ---
        recency_w = 1.0
        try:
            if hasattr(self, "_get_recency_weights") and regime is not None:
                # Ask for a vector with k_eff points
                recency_w = self._get_recency_weights(regime, n_points=len(kernel_w))
        except Exception:
            recency_w = 1.0

        # Normalize to a length-k_eff vector
        if np.isscalar(recency_w):
            recency_w = np.ones_like(kernel_w, dtype=np.float32) * float(recency_w)
        else:
            recency_w = np.asarray(recency_w, dtype=np.float32).ravel()
            if recency_w.size != kernel_w.size:
                # Interpolate/tile to match k_eff
                if recency_w.size >= 2:
                    xp = np.linspace(0.0, 1.0, recency_w.size)
                    xq = np.linspace(0.0, 1.0, kernel_w.size)
                    recency_w = np.interp(xq, xp, recency_w).astype(np.float32)
                else:
                    recency_w = np.ones_like(kernel_w, dtype=np.float32) * float(
                        recency_w[0] if recency_w.size else 1.0)

        # --- Final neighbor weights ---
        assert kernel_w.shape == outlier_mult.shape, "kernel/outlier mismatch"
        assert recency_w.shape == kernel_w.shape, f"recency shape {recency_w.shape} ≠ k_eff {kernel_w.shape}"

        w = kernel_w * recency_w * outlier_mult'''

        # --- Recompute kernel AFTER all masks so lengths match -----------------
        kernel_w = np.exp(-0.5 * dist2 / (self.h ** 2)).astype(float)  # length == len(dist2) == len(idx)

        # --- Outlier down-weighting (compute AFTER final idx is fixed) ---------
        zcap = getattr(self, "outlier_z_cap", 3.0)
        if getattr(self, "_scale_vec", None) is not None and self._scale_vec.size == self.centers.shape[1]:
            diff = (self.centers[idx] - x) / (self._scale_vec + 1e-12)
            zmax = np.max(np.abs(diff), axis=1)
            outlier_mult = np.where(zmax > zcap, np.exp(-0.5 * (zmax - zcap)), 1.0).astype(float)
        else:
            outlier_mult = np.ones_like(kernel_w, dtype=float)

        # --- Recency weights (force same length as kernel_w) -------------------
        try:
            recency_w = self._get_recency_weights(regime, n_points=len(kernel_w))
        except Exception:
            recency_w = 1.0

        if np.isscalar(recency_w):
            recency_w = np.ones_like(kernel_w, dtype=float) * float(recency_w)
        else:
            recency_w = np.asarray(recency_w, dtype=float).ravel()
            if recency_w.size != kernel_w.size:
                # resample to match
                xp = np.linspace(0.0, 1.0, max(2, recency_w.size))
                xq = np.linspace(0.0, 1.0, kernel_w.size)
                recency_w = np.interp(xq, xp, recency_w).astype(float)

        # --- Final neighbor weights (all same shape) ---------------------------
        # (Optional debug assertions while iterating)
        # assert kernel_w.shape == dist2.shape
        # assert kernel_w.shape == outlier_mult.shape
        # assert kernel_w.shape == recency_w.shape

        w = kernel_w * recency_w * outlier_mult
        ws = float(w.sum())
        w = (w / ws) if ws > 1e-12 else (np.ones_like(w) / max(1, w.size))

        w_sum = float(w.sum())
        w = (w / w_sum) if w_sum > 1e-12 else (np.ones_like(w) / max(1, w.size))

        if w_sum <= 1e-12:
            # degenerate case: fall back to uniform
            w = np.ones_like(w) / max(1, w.size)
        else:
            w = w / w_sum

        mu_k = float(np.dot(w, self.mu[idx]))
        var_k = float(np.dot(w, self.var[idx]))
        var_down_k = float(np.dot(w, self.var_down[idx]))

        print(f"[EV] KNN k={idx.size}  ids={idx[:5].tolist()}  d2={np.round(dist2[:5], 6).tolist()}")
        print(f"[EV] kernel μ={mu_k:.6f}  σ²={var_k:.6f}")

        # --- Probability (single source of truth) ------------------------------
        # Precedence: isotonic (if present) -> else kernel monotone map.
        '''try:
            if self._calibrator is not None:
                # Use the same mapping as training (isotonic over µ)
                from prediction_engine.calibration import map_mu_to_prob
                p_up = float(map_mu_to_prob(np.array([mu_k], dtype=float),
                                            calibrator=self._calibrator)[0])
                
                
                '''
        '''try:
            if self._calibrator is not None:
                #Use the same mapping as training (isotonic over µ) — use the module-level import
                p_up = float(map_mu_to_prob(np.array([mu_k], dtype=float),
                calibrator = self._calibrator)[0])

                p_source = "isotonic"
            else:
                # Monotone kernel fallback (tanh on µ magnitude)
                scale = max(1e-6, np.median([abs(mu_k), 1e-4]))
                p_up = float(0.5 * (1.0 + np.tanh(mu_k / (5.0 * scale))))
                p_source = "kernel"
        except Exception:
            p_up, p_source = 0.5, "fallback"

        p_up = float(np.clip(p_up, 1e-3, 1.0 - 1e-3))
        print(f"[Cal] p_up={p_up:.3f} source={p_source}")'''

        # --- NEW: kernel probability as a rock-solid fallback ---
        # Probability that next-bar return is > 0 using the same kernel weights.
        # y should be a 1D array of realized next-bar returns for neighbors.
        '''y = np.asarray(self.mu[idx], dtype=float) # CORRECTED: Use neighbor returns from self.mu[idx]
        w = np.asarray(w, dtype=float)
        w_sum = float(w.sum()) if w.size else 0.0

        if w_sum > 1e-9:  # Use a small epsilon for float comparison
            p_kernel = float(np.sum(w * (y > 0)) / w_sum)
        else:
            p_kernel = 0.5
            # keep p away from exact 0/1
        p_kernel = float(np.clip(p_kernel, 1e-3, 1.0 - 1e-3))

        # --- Try calibrated probability if provided; fallback to kernel ---
        p_source = "kernel"
        p_up = p_kernel
        if getattr(self, "_calibrator", None) is not None:  # CORRECTED: Use self._calibrator
            try:
                # Calibrator expects X sorted/in-range; we pass the kernel mu 'mu_k'
                p_iso = float(
                    self._calibrator.predict(np.array([mu_k], dtype=float).reshape(-1, 1))[0])  # CORRECTED: Use mu_k
                if np.isfinite(p_iso):
                    p_up = float(np.clip(p_iso, 1e-3, 1.0 - 1e-3))
                    p_source = "isotonic"
            except Exception:
                # Keep kernel fallback
                p_up = p_kernel
                p_source = "kernel"'''
        # --- Probability (single source of truth) ---
        # y_i should be your neighbor outcomes; if you store neighbor returns in self.mu, reuse them
        y_neighbors = np.asarray(self.mu[idx], dtype=float)  # or whatever array holds neighbor outcomes
        p_kernel = float(np.clip(np.sum(w * (y_neighbors > 0)) / max(1e-12, w.sum()), 1e-3, 1 - 1e-3))
        p_up, p_source = p_kernel, "kernel"

        # Optional: if your calibrator maps mu_k (local mean return) → probability, compute mu_k first:
        mu_k = float(np.sum(w * y_neighbors))  # local mean; keep for logging and/or calibrator

        if getattr(self, "_calibrator", None) is not None:
            try:
                # If your calibrator expects mu_k → p, feed mu_k; otherwise feed p_kernel.
                # Choose **one** and be consistent across training & inference.
                p_iso = float(self._calibrator.predict(np.array([[mu_k]], dtype=float))[0])
                if np.isfinite(p_iso):
                    p_up = float(np.clip(p_iso, 1e-3, 1 - 1e-3))
                    p_source = "isotonic"
            except Exception:
                pass

        # (Optional) print or log source for debugging
        # self._dbg(f"[Cal] p_up={p_up:.3f} source={p_source} mu_k={mu_k:.5f}")

        # Debug visibility
        print(f"[Cal] p_up={p_up:.3f} source={p_source}")

        # --- COST handling (unit-safe) ----------------------------------------
        mu_raw = float(mu_k)  # kernel µ before costs
        mu_net = mu_raw

        if self._cost is not None:
            # You’ll want to pass a price in the future; until then, keep costs zero in debug.
            price = getattr(self, "last_price", None)
            if price is not None and price > 0:
                usd_ps_oneway = float(self._cost.estimate())  # $/share (one-way)
                cost_frac = (usd_ps_oneway / float(price)) * 2.0  # round-trip fraction
                mu_net = mu_raw - cost_frac
                print(f"[Cost] mu_raw={mu_raw:+.6f} usd_ps={usd_ps_oneway:.4f} price={price:.4f} "
                      f"cost_frac_rt={cost_frac:.6f} mu_net={mu_net:+.6f}")
            else:
                # No price available → skip cost until wired. Debug prints to avoid confusion.
                print(f"[Cost] mu_raw={mu_raw:+.6f} (no price; skipping cost) mu_net={mu_net:+.6f}")
        else:
            print(f"[Cost] mu_raw={mu_raw:+.6f} (no cost model) mu_net={mu_net:+.6f}")

        '''# --- COST handling (unit-safe) ----------------------------------------
        # If a cost model is present, it should return $/share (one-way).
        # Convert to a *fraction of price* and subtract as round-trip.
        cost_frac = 0.0
        price_used = np.nan
        if self._cost_model is not None:
            try:
                # You can customize how we fetch a price; for O->C we use next open.
                # If price is not injected here, fall back to |mu_k| scaling or skip.
                # Minimal, safe default for now: skip unless caller later wires price.
                est_cost_ps = float(self._cost_model.estimate())  # $/share, one-way
                # Your engine doesn’t carry price at this point; until you wire it,
                # keep costs OFF in debug runs (see run_backtest change below).
                # To fully enable: pass price into evaluate(...) or set self.last_price.
                price_used = float("nan")
                cost_frac = 0.0  # keep zero until price wiring is added
            except Exception:
                cost_frac = 0.0

        # Apply cost as round-trip fraction
        mu_raw = float(mu_k)
        mu_net = float(mu_raw - 2.0 * cost_frac)

        print(
            f"[Cost] mu_raw={mu_raw:+.6f}  cost_ps_usd={getattr(self._cost_model, 'commission', float('nan')) if self._cost_model else 0.0:.4f} "
            f"price={price_used}  cost_frac_rt={2.0 * cost_frac:.6f}  mu_net={mu_net:+.6f}")
        '''
        # -----------------------------------------------------------------
        # NEW · decide whether to call analogue_synth
        # -----------------------------------------------------------------
        nearest_d2 = float(dist2.min())
        use_synth = nearest_d2 > self.tau_dist
        if use_synth:
            (mu_syn, var_syn, var_down_syn,
            beta, residual) = self._synth_cache(
            tuple(x), tuple(idx.tolist()), mu_k, var_k, var_down_k
                                                                       )
        else:
            beta = np.zeros_like(idx, dtype=np.float32)
            beta[0] = 1.0
            #residual = np.inf
            #residual = 0
            # ── FIX: keep a usable residual even without synth ──────────
            residual = nearest_d2  # <<<<<<<<< was always 0

            mu_syn, var_syn, var_down_syn = mu_k, var_k, var_down_k

        mu_final = float(mu_net)  # use cost-adjusted µ for decisions

        # ... your sizing code ...
        # kelly = self._sizer.size(mu_final, var_down_syn)   # example
        # position_size = max(0.0, min(kelly, 0.4))          # cap 40%



        '''# ------------------------------------------------------------
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
            var_down_syn = float(beta @ self.var_down[idx])'''


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

        '''res = EVResult(
            mu=mu_final, sigma=float(var_syn), variance_down=float(var_down_syn),
            beta=beta.astype(np.float32), residual=float(residual),
            cluster_id=int(idx[0]), position_size=float(0.0),
            mu_raw=mu_raw, p_up=float(p_up), p_source=p_source
        )'''



        # ------------------------------------------------------------
        # 4. Subtract transaction cost
        # ------------------------------------------------------------
        #cost_ps = self._cost.estimate(half_spread=half_spread, adv_percentile=adv_percentile)
        #print(f"[Cost] raw_mu={mu:+.6f}  cost_ps={cost_ps:+.6f}  mu_net_preCal={mu - cost_ps:+.6f}")

        # 5. ISO‑CALIBRATION  (post‑cost, pre‑return)
        '''mu_cal = (
            self._calibrator.predict([[mu]])[0]  # shape (N,1) expected
            if self._calibrator is not None
            else mu
        )'''

        # 5. ISO‑CALIBRATION  (pre‑cost): map raw µ → calibrated µ
        #mu_cal = (
        #    self._calibrator.predict([[mu]])[0]
        #    if self._calibrator is not None
        #    else mu
               # )

        #if self._calibrator is not None:
        #    mu_cal = float(self._calibrator.predict([[mu]])[0])
        #    print(f"[Cal] raw_mu={mu:+.6f}  mu_cal={mu_cal:+.6f}")
        #else:
        #    mu_cal = mu

        # ── A: override raw µ with calibrated µ ─────────────────
        #mu = mu_cal
        #mu_net = mu - cost_ps

        # 5. ISO-CALIBRATION  **before** cost
        '''if self._calibrator:
            mu = float(self._calibrator.predict([[mu]])[0])  # calibrated µ
            print(f"[Cal] µ_cal={mu:+.6f}")
        # 6. Subtract estimated cost
        #cost_ps = self._cost.estimate(half_spread=half_spread, adv_percentile = adv_percentile)
        # apply extra slippage buffer of +0.0001
        #cost_ps = self._cost.estimate(half_spread=half_spread, adv_percentile=adv_percentile)
        #cost_ps += 0.0001

        cost_ps = self._cost.estimate(
            half_spread = half_spread,  # if None, model uses 15 bps fallback
            adv_percentile = adv_percentile)

        mu_net = mu - cost_ps

        mu_net_original = mu - cost_ps
        mu_net = mu_net_original'''
        # ------------------------------------------------------------
        # 4. Keep RAW µ for trading math; do NOT overwrite with calibration
        # ------------------------------------------------------------
        mu_raw = mu  # <- preserve the model’s raw expected edge


        # 5) Transaction cost and net edge (from RAW µ)
        '''cost_ps = self._cost.estimate(
            half_spread=half_spread,
            adv_percentile=adv_percentile
        )
        mu_net = mu_raw - cost_ps
        '''

        '''if self._cost is not None:
            cost_ps = self._cost.estimate(
            half_spread = half_spread,
            adv_percentile = adv_percentile
                                        )
        else:
            cost_ps = 0.0
            mu_net = mu_raw - cost_ps

        mu_net_original = mu_net  # keep a copy before residual penalties
        '''

        if self._cost is not None:
            cost_ps = self._cost.estimate(
            half_spread = half_spread,
            adv_percentile = adv_percentile
                                       )
        else:
            cost_ps = 0.0
            mu_net = mu_raw - cost_ps
            mu_net_original = mu_net  # keep a copy before residual penalties

        # ------------------------------------------------------------
        # QUICK-WIN : penalise distant analogues
        # ------------------------------------------------------------
        '''hard_limit = self.residual_threshold * 0.66
        if residual > hard_limit:
            mu_net = 0.0
        else:
            # always shrink by 1/(1+residual), but don’t zero
            mu_net *= 1.0 / (1.0 + residual)

        # soft recovery for moderate residuals that got zeroed:
        if mu_net == 0.0 and residual <= self.residual_threshold:
            mu_net = mu_net + (mu_net_original if 'mu_net_original' in locals() else 0.0) * 0.1'''

        # replace the residual penalty block with a gentle taper
        shrink = 1.0 / (1.0 + residual / max(self.residual_threshold, 1e-6))
        mu_net *= shrink
        # DO NOT set mu_net = 0.0 on “hard_limit” – remove that branch entirely for this test

        # ------------------------------------------------------------
        # 6) Calibrated probability (single source of truth) — do NOT mutate µ
        # ------------------------------------------------------------
        '''p_up_arr = map_mu_to_prob(
            mu_raw,  # <-- use RAW µ, not mu_net
            calibrator=self._calibrator,
            #artefact_dir=self.calibration_dir,  # or where iso_calibrator.pkl lives
            default_if_missing=0.5  # NEUTRAL, not 0.6
            #artefact_dir=self.calibration_dir  # we already passed a calibrator
        )
        #p_up = float(np.atleast_1d(p_up_arr)[0])
        # after you compute mu_val (a float)
        p_up = float(
            map_mu_to_prob([mu_val], calibrator=self.calibrator, default_if_missing=0.5)[0]
        )'''

        '''p_up_arr = map_mu_to_prob(
            mu_raw,  # raw μ (float is fine)
            calibrator=self._calibrator,  # use the injected calibrator
            default_if_missing=0.5  # neutral if none available
        )
        p_up = float(np.atleast_1d(p_up_arr)[0])

        print(f"[Cal] p_up={p_up:.3f}")'''


        # (Optional: print both raw µ and calibrated mapping without changing µ)
        '''if self._calibrator:
            mu_cal = float(np.atleast_1d(self._calibrator.predict(np.atleast_1d(mu_raw)))[0])
            print(f"[Cal] raw_mu={mu_raw:+.6f}  mu_cal={mu_cal:+.6f}")
        '''
        if self._calibrator is not None:
            mu_cal = float(np.atleast_1d(self._calibrator.predict(np.atleast_1d(mu_raw)))[0])
            print(f"[Cal] raw_mu={mu_raw:+.6f}  mu_cal={mu_cal:+.6f}")

        # ------------------------------------------------------------
        # 7) Kelly position size (based on mu_net, as before)
        # ------------------------------------------------------------
        pos_size = self._sizer.size(mu_net, sig2_down ** 0.5, adv_percentile)

        # ------------------------------------------------------------
        # Drift-monitor logging uses the calibrated probability
        # ------------------------------------------------------------
        from math import erf, sqrt as _sqrt  # (kept if you use erf elsewhere)

        dm = get_monitor()
        ticket_id = dm.log_pred(p_up)
        if dm.status()[0] is DriftStatus.RETRAIN_SIGNAL:
            from threading import Thread
            Thread(target=_kickoff_retrain, daemon=True).start()

        print(f"[EV] DONE  mu_net={mu_net:.6f}  cluster_id={int(idx[0])}  pos_size={pos_size:.4f}")
        print(f"[Cost] raw_mu={mu_raw:+.6f}  cost_ps={cost_ps:+.6f}  mu_net={mu_net:+.6f}")
        # ------------------------------------------------------------------
        # QUICK-WIN : penalise distant analogues
        # • Hard gate  : skip anything whose residual > τ₇₅/3
        # • Soft weight: shrink edge by 1/(1+residual)
        # ------------------------------------------------------------------

        #if residual > (self.residual_threshold / 3.0):
        #    mu_net = 0.0  # treat as no-edge
        # allow further-out analogues by raising threshold

        '''if residual > (self.residual_threshold * 2.0):
            mu_net = 0.0
        else:
            mu_net *= 1.0 / (1.0 + residual)'''

        # softened gate on residual: allow up to half threshold
        '''hard_limit = self.residual_threshold / 2.0
        if residual > hard_limit:
            mu_net = 0.0
        else:
            mu_net *= 1.0 / (1.0 + residual * 0.5)  # softer shrink
        '''

        '''hard_limit = self.residual_threshold * 0.66
        if residual > hard_limit:
            mu_net = 0.0
        else:
            # always shrink by 1/(1+residual), but don’t zero
            mu_net *= 1.0 / (1.0 + residual)'''



        #mu_net = mu - cost_ps
        #mu = mu_cal  # use calibrated value
        #mu_net = mu - cost_ps

        #if self._calibrator:
        #    mu_cal = float(self._calibrator.predict([[mu]])[0])
        #    print(f"[Cal] raw_mu={mu:+.6f}  mu_cal={mu_cal:+.6f}")

        '''# (old) we already zeroed mu_net on high residual
        # now even if we zero above, add a soft fallback:
        if mu_net == 0.0 and residual <= self.residual_threshold:
            # recover a bit of edge for moderate residuals
            mu_net = mu_net + (mu_net_original if 'mu_net_original' in locals() else mu_net) * 0.1



        # 6. if you want P(win), _compute it here_ but keep µ_net pure:
        p_win = (self._calibrator.predict([[mu]])[0]
                 if self._calibrator else None)
'''
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


        '''# ------------------------------------------------------------
        # 6. Kelly position size (NEW)
        # ------------------------------------------------------------
        pos_size = self._sizer.size(mu_net, sig2_down ** 0.5, adv_percentile)

        # ------------------------------------------------------------
        # 7. Drift-monitor registration  (prob ≈ P(μ>0) assuming normal)
        # ------------------------------------------------------------
        from math import erf, sqrt as _sqrt


        #p_up = 0.5 * (1.0 + erf(mu / (_sqrt(2.0 * sig2) + 1e-9)))

        # --- Calibrated probability for logging/monitoring (single source of truth)
        p_up_arr = map_mu_to_prob(
            mu,
            calibrator=self._calibrator,
            artefact_dir=None  # engine already has self._calibrator if available
        )
        p_up = float(np.atleast_1d(p_up_arr)[0])
        print(f"[Cal] p_up={p_up:.3f}")'''


        '''dm = get_monitor()
        ticket_id = dm.log_pred(p_up)  # OMS must later call
        # dm.log_outcome(ticket_id, realised_pnl > 0)

        if dm.status()[0] is DriftStatus.RETRAIN_SIGNAL:
            from threading import Thread
            Thread(target=_kickoff_retrain, daemon=True).start()

        print(f"[EV] DONE  mu_net={mu_net:.6f}  cluster_id={int(idx[0])}  pos_size={pos_size:.4f}")
        # ────────────────────────────────────────────────────────────────────

        print(f"[Cost] raw_mu={mu:+.6f}  cost_ps={cost_ps:+.6f}  "
              f"mu_net={mu_net:+.6f}")'''

        return EVResult(
            mu_net,  # mu
            sig2,  # sigma
            sig2_down,  # variance_down
            beta,  # beta array
            residual,  # residual
            int(idx[0]),  # cluster_id
            blended_probs,  # outcome_probs
            pos_size,  # position_size
            ticket_id,  # drift_ticket
            mu_raw=mu_raw,
            p_up=p_up,
            p_source=p_source,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha1_list(items: Iterable[str]) -> str:

    feat_str = "|".join(items)
    full = hashlib.sha1(feat_str.encode()).hexdigest()
    return full[:12]
