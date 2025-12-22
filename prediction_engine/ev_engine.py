# ---------------------------------------------------------------------------
# prediction_engine/ev_engine.py  –  k capped by centres AND feature‐dimension
# ---------------------------------------------------------------------------
from __future__ import annotations

import hashlib

import pandas as pd
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
from typing import Iterable, Literal, Tuple, Dict, Any, Optional
import joblib
from prediction_engine.position_sizer import KellySizer        # NEW
from .drift_monitor import get_monitor, DriftStatus        # NEW
from typing import Iterable, Literal, Tuple, Dict, Any
from functools import lru_cache
from typing import Mapping, Any
import dataclasses

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

# --- begin: curve param coercion helper (add after CurveParams import) ---
from dataclasses import fields as _dc_fields

# detect the tail field name in CurveParams
_TAIL_FIELD = next((f.name for f in _dc_fields(CurveParams) if f.name in ("tail_len", "tail")), None)



# ADD ▼▼▼ regime keys and adjacency map
_REGIME_ORDER = ("TREND", "RANGE", "VOL", "GLOBAL")
_ADJACENT = {
    "TREND":  ("RANGE", "VOL"),
    "RANGE":  ("TREND", "VOL"),
    "VOL":    ("RANGE", "TREND"),
    "GLOBAL": tuple(),
}



def _coerce_curve_params_dict(raw: Mapping[str, Any]) -> dict[str, Any]:
    """
    Normalise a raw JSON payload into kwargs suitable for CurveParams.

    Handles both:
      * {"params": {...}} payloads (newer style)
      * flat {"family": "...", "tail_len_days": 20, "alpha": 1.5, ...}
    and a few legacy field names.
    """
    # Don't ever early-return an empty dict; CurveParams needs required fields.
    if raw is None:
        raw = {}

    # Unwrap {"params": {...}} if present
    if "params" in raw and isinstance(raw["params"], dict):
        inner = raw["params"]
    else:
        inner = raw

    # Lower-case keys for robustness
    norm: dict[str, Any] = {str(k).lower(): v for k, v in inner.items()}

    # Determine which field in CurveParams represents "tail length"
    tail_field = None
    for f in dataclasses.fields(CurveParams):
        if f.name in ("tail_len", "tail", "tail_len_days"):
            tail_field = f.name
            break
    if tail_field is None:
        tail_field = "tail_len"

    aliases: dict[str, str] = {
        # Tail length variants
        "tail_len_days": tail_field,
        "tail_len": tail_field,
        "tail": tail_field,

        # Main fields we expect to keep as-is
        "family": "family",
        "shape": "shape",
        "shape_param": "shape",   # legacy
        "shape_params": "shape",  # legacy

        # Legacy "alpha" knobs → blend_alpha
        "alpha": "blend_alpha",
        "a": "blend_alpha",
        "blend_alpha": "blend_alpha",

        # Lambda regularisation
        "lambda": "lambda_reg",
        "lambda_reg": "lambda_reg",
    }

    wanted = {f.name for f in dataclasses.fields(CurveParams)}
    out: dict[str, Any] = {}

    # Map raw keys → canonical CurveParams field names
    for k, v in norm.items():
        key = aliases.get(k, k)
        if key in wanted and v is not None:
            out[key] = v

    # ---------- REQUIRED FIELDS: provide safe defaults ----------

    # Tail length default if none was supplied
    if tail_field in wanted and tail_field not in out:
        # Reasonable default tail length in days
        out[tail_field] = 20

    # Family default
    if "family" not in out and "family" in wanted:
        out["family"] = "exp"

    # ---------- OPTIONAL FIELDS: safe defaults ----------

    if "shape" in wanted and "shape" not in out:
        # Neutral-ish shape – will behave close to flat / linear
        out["shape"] = 1.0

    if "blend_alpha" in wanted and "blend_alpha" not in out:
        # You can change this to 0.25 if you want to match Phase 2 exactly
        out["blend_alpha"] = 0.5

    if "lambda_reg" in wanted and "lambda_reg" not in out:
        out["lambda_reg"] = 1.0

    return out




# --- ADD: lightweight ANN index artifact helpers ---
from datetime import datetime
def _load_json(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}

def _save_json(p: Path, obj: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True))

def _sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()[:12]


__all__: Tuple[str, ...] = ("EVEngine", "EVResult")




# ev_engine.py — near the top, after imports/config
from dataclasses import dataclass

@dataclass
class SynthGateParams:
    residual_tau: float = 0.25         # conservative until learned from validation
    min_beta_nz: int = 2               # require at least 2 active betas
    sign_flip_gain: float = 1.5        # if flipping sign, require >=1.5x magnitude vs kernel
    prefer_kernel_when_tied: bool = True

GATE = SynthGateParams()


# ev_engine.py — after the SynthGateParams dataclass and GATE assignment

# at top
from typing import Optional

def _beta_nz(beta: Optional[np.ndarray], eps: float = 1e-6) -> int:
    if beta is None:
        return 0
    return int(np.count_nonzero(beta > eps))


#def _beta_nz(beta: np.ndarray, eps: float = 1e-6) -> int:
#    return int(np.count_nonzero(beta > eps))


# ---------------------------------------------------------------------------
# Public dataclass returned to ExecutionManager
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class EVResult:
    # core outputs
    mu: float                      # final expected return per share (after path choice)
    sigma: float                   # final variance (σ²)
    variance_down: float           # downside variance (Sortino / Kelly)

    # synthesis diagnostics
    residual: float
    beta: Optional[NDArray[np.float32]] = field(default=None, repr=False)
    path: str = "kernel"           # "synth" | "kernel"
    k_used: int = 0
    metric: str = "euclidean"

    # cluster / outcomes / sizing
    cluster_id: int = -1
    outcome_probs: Dict[str, float] = field(default_factory=dict)
    position_size: float = 0.0

    # drift / calibration
    drift_ticket: int = -1
    mu_raw: float = 0.0
    p_up: float = 0.5
    p_source: str = "kernel"


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
        _center_age_days: np.ndarray | None = None,

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

        # ADD ▼▼▼ diagnostics containers
        self._fallback_counts = {k.lower(): 0 for k in _REGIME_ORDER}
        self._search_log = []  # list of dicts per evaluate() call

        # NEW: Store regime-specific curves and outcome probabilities
        self.regime_curves = regime_curves or {}
        self.outcome_probs = outcome_probs or {}
        self._center_age_days = (
            np.asarray(_center_age_days, dtype=np.float32) if _center_age_days is not None else None
        )

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

        '''self._dist = DistanceCalculator(
            ref = self.centers,
            metric = metric,
            rf_weights = rf_weights,
        )'''

        # --- REPLACE the existing DistanceCalculator construction with: ---
        self._dist = DistanceCalculator(
            ref=self.centers,
            metric=metric,
            rf_weights=rf_weights,
        )
        # best-effort: if the calculator exposes a backend name, keep it for logs
        self._index_backend = getattr(self._dist, "_ann_backend", "sklearn" if metric == "euclidean" else "none")

        #self._cost: BasicCostModel = cost_model or BasicCostModel()

        # --- ADD: remember which index backend the distance helper is using
        #self._index_backend = "none"

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


            '''if residual > self.residual_threshold:  # kernel fallback
                beta[:] = 0.0
                beta[0] = 1.0
                mu_syn, var_syn, var_down_syn = mu_k, var_k, var_down_k
            else:
                mu_syn = float(beta @ self.mu[idx])
                var_syn = float(beta @ self.var[idx])
                var_down_syn = float(beta @ self.var_down[idx])
            '''

            '''mu_syn = float(beta @ self.mu[idx])
            var_syn = float(beta @ self.var[idx])
            var_down_syn = float(beta @ self.var_down[idx])
            '''

            # Means combine linearly, variances combine with β² assuming independence
            mu_syn = float(beta @ self.mu[idx])
            var_syn = float((beta ** 2) @ self.var[idx])
            var_down_syn = float((beta ** 2) @ self.var_down[idx])

            print(f"[EV] synth residual={residual:.5g}  α={self.alpha:.2f}")
            print(f"[EV] β (first 5)={np.round(beta[:5], 4).tolist()}")
            print(f"[EV] synth μ={mu_syn:.6f}  σ²={var_syn:.6f}")

            return mu_syn, var_syn, var_down_syn, beta.astype(np.float32, copy=False), residual

    @classmethod
    def from_artifacts(
        cls,
        artefact_dir: Path | str,
        *,
        #metric: Literal["euclidean", "mahalanobis", "rf_weighted"] = "euclidean",
        metric: Literal["euclidean", "mahalanobis", "rf_weighted"] = "rf_weighted",
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

        # --- NEW: optional centroid ages/dates ---------------------------------
        age_days_file = artefact_dir / "center_age_days.npy"
        date_file = artefact_dir / "center_dates.npy"

        center_age_days = None
        if age_days_file.exists():
            center_age_days = np.load(age_days_file).astype(np.float32)
        elif date_file.exists():
            # Compute age in days vs file mtime (proxy) or “trained_on_date” if you prefer
            dates = np.load(date_file)
            # Safe fallback: treat all ages as equal if dates are missing/non-parsable
            try:
                # Use artifact meta trained_on_date if present
                meta_path = artefact_dir / "meta.json"
                ref_ts = None
                if meta_path.exists():
                    _meta = json.loads(meta_path.read_text())
                    ref_ts = pd.Timestamp(_meta.get("trained_on_date"))
                ref_ts = ref_ts or pd.Timestamp.utcnow()
                center_age_days = (ref_ts - pd.to_datetime(dates)).days.astype(np.float32)
            except Exception:
                center_age_days = None

        # Store on engine (even if None)

        regime_arr = stats["regime"] if "regime" in stats.files else np.full(len(stats["mu"]), "ANY", dtype="U10")

        # ─── load all per‐regime curves ────────────────────
        from prediction_engine.weight_optimization import load_regime_curves
        weight_root = Path("artifacts") / "weights"
        curves = {}
        if weight_root.exists():
            curves = load_regime_curves(weight_root)


        # NEW: Load all available regime-specific weighting curves
        '''regime_curves: dict[str, CurveParams] = {}
        for regime_dir in artefact_dir.glob("regime=*"):
            regime_name = regime_dir.name.split("=")[-1]
            params_file = regime_dir / "curve_params.json"
            if params_file.exists():
                params_data = json.loads(params_file.read_text())["params"]
                regime_curves[regime_name] = CurveParams(**params_data)
        '''

        # --- Load per-regime recency curves (accept both flat and {"params": {...}}) ---
        # --- Load per-regime recency curves (tolerant loader) ---
        regime_curves: dict[str, CurveParams] = {}
        for regime_dir in artefact_dir.glob("regime=*"):
            regime_name = regime_dir.name.split("=")[-1]
            params_file = regime_dir / "curve_params.json"
            if not params_file.exists():
                continue
            raw = json.loads(params_file.read_text())
            try:
                kwargs = _coerce_curve_params_dict(raw)
                regime_curves[regime_name] = CurveParams(**kwargs)
            except Exception as e:
                log.warning(
                    "[EVEngine] Failed to parse curve_params for regime=%s at %s: %s. "
                    "Using uniform weights for this regime.",
                    regime_name, str(params_file), e
                )

        # -------- sanity-check required arrays --------------------

        for key in ("mu", "var", "var_down"):
            if key not in stats.files:  # older file from previous build
                raise RuntimeError(
                    f"{stats_path} is missing '{key}'. "
                    "Delete weights directory and rebuild centroids."
                )


        #h = float(json.loads((artefact_dir / "kernel_bandwidth.json").read_text())["h"])

        # Schema guard – fail fast if feature list drifted
        '''meta = json.loads((artefact_dir / "meta.json").read_text())
        features_live = stats["feature_list"].tolist()

        print("[DEBUG] meta['sha']:", meta["sha"])
        print("[DEBUG] current sha:", _sha1_list(features_live))
        print("[DEBUG] meta['features']:", meta.get("features"))
        print("[DEBUG] current features:", features_live)

        if meta["sha"] != _sha1_list(features_live):            raise RuntimeError(
                "Feature schema drift detected – retrain PathClusterEngine before loading EVEngine."
            )'''

        meta = json.loads((artefact_dir / "meta.json").read_text())
        features_live = stats["feature_list"].tolist()
        # NOTE: feature-schema validation is handled against feature_schema.json below.

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

        # --- 3.4: enforce distance contract ---------------------------------
        meta_dist = (meta.get("payload") or meta).get("distance", {})  # tolerate legacy shape
        meta_family = str(meta_dist.get("family", metric)).lower()
        if str(metric).lower() != meta_family:
            raise RuntimeError(
                f"Metric mismatch: requested '{metric}' but artifacts were built with '{meta_family}'. "
                "Rebuild artifacts or request the same metric."
            )

        # Validate Mahalanobis contract if chosen
        if meta_family == "mahalanobis":
            cov_path = artefact_dir / "centroid_cov_inv.npy"
            if not cov_path.exists():
                raise FileNotFoundError("Mahalanobis selected but centroid_cov_inv.npy missing")
            # Optional: verify SHA1 matches meta
            '''want = (meta_dist.get("params") or {}).get("cov_inv_sha1")
            if want:
                import hashlib
                have = hashlib.sha1(cov_path.read_bytes()).hexdigest()[:12]
                if have != want:
                    raise RuntimeError(f"Mahalanobis cov_inv sha mismatch: {have} != {want}")
            '''

            want = (meta_dist.get("params") or {}).get("cov_inv_sha1")
            if want:
                import hashlib, numpy as _np
                cov_inv_arr = _np.load(cov_path).astype(_np.float32, copy=False)
                have = hashlib.sha1(cov_inv_arr.tobytes()).hexdigest()[:12]
                if have != want:
                    raise RuntimeError(f"Mahalanobis cov_inv sha mismatch: {have} != {want}")
        # Validate RF-weighted contract if chosen
        if meta_family == "rf_weighted":
            rf_path = artefact_dir / "rf_feature_weights.npy"
            if not rf_path.exists():
                raise FileNotFoundError("rf_weighted selected but rf_feature_weights.npy missing")
            import numpy as _np, hashlib
            w = _np.load(rf_path, allow_pickle=False).astype(_np.float32, copy=False)
            if w.ndim != 1 or w.shape[0] != n_feat_schema:
                raise RuntimeError(
                    f"RF weights length {w.shape} does not match feature schema ({n_feat_schema})"
                )
            # Normalise for deterministic scaling
            s = float(w.sum());
            if s <= 0 or not _np.isfinite(s):
                raise RuntimeError("RF weights sum is non-positive or NaN")
            w /= s
            # Optional SHA check
            want = (meta_dist.get("params") or {}).get("rf_weights_sha1")
            if want:
                have = hashlib.sha1(w.tobytes()).hexdigest()[:12]
                if have != want:
                    raise RuntimeError(f"RF weights sha mismatch: {have} != {want}")
            rf_w = w  # hand to DistanceCalculator ctor below
        else:
            rf_w = None
        # --------------------------------------------------------------------



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

        #cov_inv = None
        #rf_w = None
        # --- ADD: RF-weighted metric guards and normalization ---
        # --- ADD: Mahalanobis covariance sanity (optional) ---
        cov_inv = None
        if metric == "mahalanobis":
            cov_path = artefact_dir / "centroid_cov_inv.npy"
            if not cov_path.exists():
                raise FileNotFoundError("centroid_cov_inv.npy missing for Mahalanobis metric")
            cov_inv = np.load(cov_path).astype(np.float32)

        rf_w = None
        if metric == "rf_weighted":
            rf_path = artefact_dir / "rf_feature_weights.npy"
            if not rf_path.exists():
                raise FileNotFoundError(
                    f"rf_feature_weights.npy missing in {artefact_dir}; "
                    "train must persist RF importances next to centers."
                )
            rf_w = np.load(rf_path, allow_pickle=False).astype(np.float32)
            if rf_w.ndim != 1 or rf_w.shape[0] != n_feat_schema:
                raise RuntimeError(
                    f"RF weight length {rf_w.shape} ≠ feature schema length {n_feat_schema}"
                )
            s = float(rf_w.sum())
            if not np.isfinite(s) or s <= 0:
                raise RuntimeError("RF weights sum to non-positive; retrain weights.")
            rf_w /= s  # normalise → Σw=1

            # optional: enforce feature order integrity for RF weights, via meta
            rf_meta = _load_json(artefact_dir / "rf_weights_meta.json")
            feat_sha = _sha1_list(feature_names)
            expected_sha = rf_meta.get("feature_sha1")
            if expected_sha and expected_sha != feat_sha:
                raise RuntimeError(
                    f"RF weights feature_sha mismatch: {expected_sha} vs {feat_sha}."
                )

        #elif metric == "mahalanobis":
        #    cov_inv = np.load(artefact_dir / "centroid_cov_inv.npy")
        '''elif metric == "rf_weighted":
            rf_file = artefact_dir / "rf_feature_weights.npy"
            rf_w = np.load(rf_file, allow_pickle=False)
        '''
        #if not rf_path.exists():
        #        raise FileNotFoundError("RF weights file missing for rf_weighted metric")

        # --- ADD: ANN index contract (metadata only if no FAISS file) ---
        index_root = artefact_dir / "index_backends"
        backend = "sklearn" if metric == "euclidean" else "none"
        meta_path = index_root / backend / "meta.json"

        # create/refresh contract if missing
        if not meta_path.exists():
            _save_json(meta_path, {
                "index_type": backend,
                "metric": metric,
                "feature_sha1": _sha1_list(feature_names),
                "dim": int(centers.shape[1]),
                "trained_on_date": datetime.utcnow().isoformat(),
                "n_items": int(centers.shape[0]),
                "files": []  # fill when you add a real FAISS/Annoy index file
            })

        #return cls(
        '''engine = cls(
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
            #residual_threshold=tau_sq_p75,
            #residual_threshold=0.35,
            residual_threshold=1.5,
            _scale_vec=scale_vec,
            _n_feat_schema=n_feat_schema,
            cluster_regime=regime_arr,
        )'''

        engine = cls(
            centers=centers,
            mu=stats["mu"],
            var=stats["var"],
            var_down=stats["var_down"],
            blend_alpha=blend_alpha,
            h=h,
            _center_age_days=center_age_days,
            metric=metric,
            cov_inv=cov_inv,
            rf_weights=rf_w,
            k=k,
            outcome_probs=outcome_probs,
            regime_curves={**regime_curves, **curves},
            cost_model=cost_model,
            calibrator=calibrator,
            residual_threshold=1.5,  # keep your current gate
            _scale_vec=np.asarray(schema.get("scales", []), dtype=np.float32),
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

    def _get_recency_weights(self, regime: MarketRegime, ages_days: np.ndarray) -> np.ndarray:
        """
        Compute recency weights from *ages_days* using the active regime curve.
        If no curve, return ones.
        """
        if ages_days is None or ages_days.size == 0:
            return np.ones(0, dtype=np.float32)

        curve = (
                    self.regime_curves.get(regime.name.lower()) if regime is not None else None
                ) or self.regime_curves.get("all")

        if not curve:
            return np.ones_like(ages_days, dtype=np.float32)

        fam = getattr(curve, "family", "exp")
        # Common params (fall back safely)
        alpha = float(getattr(curve, "alpha", 1.0))
        tail = float(getattr(curve, "tail_len_days", 20.0))
        x = np.asarray(ages_days, dtype=np.float32).clip(min=0)

        if fam == "exp":
            w = np.exp(-(x / max(1e-6, tail)) * alpha)
        elif fam == "linear":
            w = np.maximum(0.0, 1.0 - (x / max(1e-6, tail)) * alpha)
        elif fam == "sigmoid":
            # center at tail/2 with slope alpha
            z = (x - 0.5 * tail) / max(1e-6, tail / max(1.0, alpha))
            w = 1.0 / (1.0 + np.exp(z))
        else:
            w = np.ones_like(x)

        # normalise but avoid all-zeros
        s = float(w.sum())
        return (w / s) if s > 1e-12 else np.ones_like(w) / max(1, w.size)

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
        '''if regime is not None:
            reg_mask = self.cluster_regime[idx] == regime.name.upper()
            if reg_mask.any():  # keep matches when possible
                idx, dist2 = idx[reg_mask], dist2[reg_mask]
            else:  # ▲ NEW soft-fallback
                log.warning(
                    "[EVEngine] regime=%s yielded zero matches; "
                    "falling back to all clusters",
                    regime.name,
                    )'''

                # fall back to full idx/dist2 already computed above

        #regime_name = getattr(regime, "name", "ANY")

        # AFTER
        # Accept MarketRegime, string, or None
        if regime is None:
            _reg_name = "GLOBAL"
        elif isinstance(regime, str):
            _reg_name = regime
        else:
            _reg_name = getattr(regime, "name", "GLOBAL")

        reg_key_upper = (_reg_name or "GLOBAL").upper()
        if reg_key_upper not in _REGIME_ORDER:
            reg_key_upper = "GLOBAL"
        reg_key_lower = reg_key_upper.lower()

        # ADD ▼▼▼ desired regime name, normalized
        reg_key = (_reg_name or "GLOBAL").upper()
        if reg_key not in _REGIME_ORDER:
            reg_key = "GLOBAL"

        #index_used = reg_key
        #fallback_depth = 0

        index_used = reg_key_upper  # keep the log consistent with existing format
        fallback_depth = 0
        candidates = [reg_key_upper] + list(_ADJACENT[reg_key_upper]) + (
            ["GLOBAL"] if reg_key_upper != "GLOBAL" else [])

        # REPLACE regime masking block ▼▼▼
        candidates = [reg_key] + list(_ADJACENT[reg_key]) + (["GLOBAL"] if reg_key != "GLOBAL" else [])
        kept = None
        for depth, cand in enumerate(candidates):
            mask = (self.cluster_regime[idx] == cand)
            if mask.any():
                idx, dist2 = idx[mask], dist2[mask]
                index_used = cand
                fallback_depth = depth
                break
        # if none matched, keep original (kernel will still run)
        if idx.size == 0:
            index_used = "GLOBAL"
            fallback_depth = len(candidates)  # “full” fallback
            idx, dist2 = idx_all[:k_eff], dist2_all[:k_eff]

        # ---------------------------------------------------------------------

        # (distance threshold still applied earlier; no need to recompute)

        # ---------------------------------------------------------------------

        # ADD ▼▼▼ diagnostics tally
        #key = (reg_key.lower())
        key = reg_key_lower

        if fallback_depth > 0:
            self._fallback_counts[key] = int(self._fallback_counts.get(key, 0)) + 1

        # Keep a small rolling log (you can trim if needed)
        try:
            med_d = float(np.median(dist2)) if idx.size else float("nan")
        except Exception:
            med_d = float("nan")

        self._search_log.append({
            "regime_requested": reg_key,
            "index_used": index_used,
            "fallback_depth": int(fallback_depth),
            "k_eff": int(idx.size),
            "median_d2": med_d,
        })

        log.info("[EV] regime=%s index_used=%s fallback_depth=%d k=%d med_d2=%.6f",
                 reg_key, index_used, fallback_depth, int(idx.size), med_d)

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

        # --- NEW: recency weights from ages (per neighbor) --------------------
        if self._center_age_days is not None:
            ages_k = self._center_age_days[idx]  # length == len(idx)
        else:
            ages_k = None

        try:
            recency_w = self._get_recency_weights(regime, ages_k) if ages_k is not None else 1.0
        except Exception:
            recency_w = 1.0

        if np.isscalar(recency_w):
            recency_w = np.ones_like(kernel_w, dtype=float) * float(recency_w)
        # shapes now: kernel_w == outlier_mult == recency_w == (k,)

        '''w = kernel_w * outlier_mult * recency_w
        ws = float(w.sum())
        w = (w / ws) if ws > 1e-12 else (np.ones_like(w) / max(1, w.size))
        '''
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

        # --- Try isotonic only if available; fall back if it looks degenerate ---
        if getattr(self, "_calibrator", None) is not None:
            try:
                p_iso = float(self._calibrator.predict(np.array([[mu_k]], dtype=float))[0])
                # Keep within (1e-3, 1-1e-3)
                p_iso = float(np.clip(p_iso, 1e-3, 1.0 - 1e-3))

                # Reject isotonic if it is near-constant or wildly off kernel
                # (heuristics; tighten as you validate)
                if (0.1 <= p_iso <= 0.9) and (abs(p_iso - p_kernel) <= 0.40):
                    p_up, p_source = p_iso, "isotonic"
                else:
                    # leave kernel
                    pass
            except Exception:
                pass
        # Optional: if your calibrator maps mu_k (local mean return) → probability, compute mu_k first:
        mu_k = float(np.sum(w * y_neighbors))  # local mean; keep for logging and/or calibrator

        '''if getattr(self, "_calibrator", None) is not None:
            try:
                # If your calibrator expects mu_k → p, feed mu_k; otherwise feed p_kernel.
                # Choose **one** and be consistent across training & inference.
                p_iso = float(self._calibrator.predict(np.array([[mu_k]], dtype=float))[0])
                if np.isfinite(p_iso):
                    p_up = float(np.clip(p_iso, 1e-3, 1 - 1e-3))
                    p_source = "isotonic"
            except Exception:
                pass'''

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
        '''nearest_d2 = float(dist2.min())
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
        '''



        # ... your sizing code ...
        # kelly = self._sizer.size(mu_final, var_down_syn)   # example
        # position_size = max(0.0, min(kelly, 0.4))          # cap 40%

        # -----------------------------------------------------------------
        # Analogue Synthesis (compute once), then select path by residual
        # -----------------------------------------------------------------
        # 1) Compute the synthetic candidate using the cached heavy math.
        mu_syn, var_syn, var_down_syn, beta, residual = self._synth_cache(
            tuple(x), tuple(idx.tolist()), mu_k, var_k, var_down_k
        )

        # --- Additional guards on top of residual threshold ---
        beta_nz = int(np.sum(beta > 1e-6)) if isinstance(beta, np.ndarray) else 0
        sign_flip = np.sign(mu_syn) != np.sign(mu_k) and (mu_k != 0.0)

        use_synth = True

        # 1) residual gate (primary)
        if residual > self.residual_threshold:
            use_synth = False

        # 2) require some sparsity: at least a couple of active neighbours
        if use_synth and beta_nz < GATE.min_beta_nz:
            use_synth = False

        # 3) protect against destructive sign flips:
        #    only allow synth to flip sign if it beats kernel magnitude by a margin
        if use_synth and sign_flip:
            if abs(mu_syn) < GATE.sign_flip_gain * abs(mu_k):
                use_synth = False

        # 4) tie-breaker: prefer kernel when nearly equal
        if use_synth and GATE.prefer_kernel_when_tied:
            if abs(mu_syn - mu_k) < 0.25 * max(1e-12, abs(mu_k)):
                use_synth = False

        final_path = "synth" if use_synth else "kernel"

        # Pick path-specific components for the rest of the computation
        mu_raw_path = mu_syn if use_synth else mu_k
        var_path = var_syn if use_synth else var_k
        var_down_path = var_down_syn if use_synth else var_down_k

        # If not using synth, drop β from the result to keep objects light
        if not use_synth:
            beta = None

        # === Residual-gated path selection (strict) ===
        nz = _beta_nz(beta)
        agree_sign = (np.sign(mu_syn) == np.sign(mu_k)) or (abs(mu_k) < 1e-12)

        # If synth flips sign, demand a margin multiplier (e.g., 1.5×)
        if np.sign(mu_syn) != np.sign(mu_k) and abs(mu_k) >= 1e-12:
            sign_ok = (abs(mu_syn) >= GATE.sign_flip_gain * abs(mu_k))
        else:
            sign_ok = True

        resid_ok = (residual <= self.residual_threshold)
        beta_ok = (nz >= GATE.min_beta_nz)
        mag_tie = (abs(mu_syn) <= 1.05 * abs(mu_k))  # ~5% tie band

        use_synth = bool(resid_ok and beta_ok and sign_ok and (agree_sign or sign_ok))

        # Prefer kernel on ties if configured
        if GATE.prefer_kernel_when_tied and use_synth and mag_tie:
            use_synth = False

        final_path = "synth" if use_synth else "kernel"

        # Choose path-specific μ/σ²
        mu_raw_path = mu_syn if use_synth else mu_k
        var_path = var_syn if use_synth else var_k
        var_down_path = var_down_syn if use_synth else var_down_k

        # If not using synth, avoid carrying β for logging payload bloat
        if not use_synth:
            beta = None

        # Always have a local numeric beta vector for downstream math
        beta_arr = beta if isinstance(beta, np.ndarray) else np.zeros(len(idx), dtype=np.float32)



        '''# 2) Decide the path by residual (low residual ⇒ accept synth)
        use_synth = bool(residual <= self.residual_threshold)
        final_path = "synth" if use_synth else "kernel"

        # 3) Choose raw μ/σ² from the chosen path
        mu_raw_path = mu_syn if use_synth else mu_k
        var_path = var_syn if use_synth else var_k
        var_down_path = var_down_syn if use_synth else var_down_k

        # If we didn’t use synth, avoid bloating logs/result with a long β vector
        if not use_synth:
            beta = None
        '''


        # 4) Compute transaction costs (reuse existing cost model if present)
        #    This block mirrors your earlier cost logic, but applies to the CHOSEN μ.
        price = getattr(self, "last_price", None)
        if self._cost is not None and price is not None and price > 0:
            try:
                usd_ps_oneway = float(self._cost.estimate())  # $/share one-way
                cost_frac_rt = (usd_ps_oneway / float(price)) * 2.0
            except Exception:
                cost_frac_rt = 0.0
        else:
            cost_frac_rt = 0.0

        mu_raw = float(mu_raw_path)  # keep raw μ from chosen path
        mu_net = float(mu_raw - cost_frac_rt)  # subtract round-trip cost (fractional)

        # 5) Gentle residual shrink (applies regardless of path)
        shrink = 1.0 / (1.0 + residual / max(self.residual_threshold, 1e-6))
        mu_net *= shrink

        # 6) Position sizing uses downside variance from the CHOSEN path
        pos_size = self._sizer.size(mu_net, float(var_down_path) ** 0.5, adv_percentile)

        # 7) Single-source-of-truth probability (you computed p_up/p_source above)
        #    (keep your existing computation of p_up / p_source just prior to cost)

        # 8) One-line diagnostic log for quick grep-based analysis
        '''log.info(
            "EV path=%s k=%d metric=%s resid=%.3e mu_k=%.6g mu_s=%.6g mu=%+.6g p=%.3f src=%s",
            final_path, int(idx.size), self.metric, residual, mu_k, mu_syn, mu_net, p_up, p_source
        )'''

        '''log.info(
            "EV path=%s k=%d metric=%s index_backend=%s resid=%.3e mu_k=%.6g mu_s=%.6g mu=%+.6g p=%.3f src=%s",
            final_path, int(idx.size), self.metric, getattr(self, "_index_backend", "none"),
            residual, mu_k, mu_syn, mu_net, p_up, p_source
        )'''

        #curve_used = (
        #                 self.regime_curves.get(regime.name.lower()) if regime is not None else None
        #             ) or self.regime_curves.get("all")

        # Prefer the requested regime if present; otherwise try "all"
        curve_used = (
                self.regime_curves.get(reg_key_lower)
                or self.regime_curves.get(reg_key_upper)
                or self.regime_curves.get("all")
                or self.regime_curves.get("global")
                or None
        )

        rec_fam = getattr(curve_used, "family", "uniform")
        rec_tail = getattr(curve_used, "tail_len_days", 0)

        log.info(
            "EV path=%s k=%d metric=%s index_backend=%s resid=%.3e mu_k=%.6g mu_s=%.6g mu=%+.6g p=%.3f src=%s "
            "regime=%s recency_family=%s tail_len_days=%s",
            final_path, int(idx.size), self.metric, getattr(self._dist, "_ann_backend", "sklearn"),
            residual, mu_k, mu_syn, mu_net, p_up, p_source,
            getattr(regime, "name", "NONE"), rec_fam, rec_tail
        )

        # We’ll carry on to outcome-prob blending (probs_k/probs_syn) which you have
        # below, then construct EVResult with the chosen path + diagnostics.

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

        #alpha = self.alpha
        #mu = alpha * mu_syn + (1 - alpha) * mu_k
        #sig2 = alpha ** 2 * var_syn + (1 - alpha) ** 2 * var_k
        #sig2_down = alpha ** 2 * var_down_syn + (1 - alpha) ** 2 * var_down_k


        #if adv_percentile is None:
        #    liquidity_mult = 1.0
        #else:  # piece‑wise linear: 1× below 5 % ADV  → 1.5× at 20 %
        #    liquidity_mult = 1.0 + 0.5 * min(max((adv_percentile - 5) / 15, 0.0), 1.0)

        # Always use flat volatility during debugging
        #liquidity_mult = 1.0

        #sig2 *= liquidity_mult
        #sig2_down *= liquidity_mult

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
        #mu_raw = mu  # <- preserve the model’s raw expected edge


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
        #pos_size = self._sizer.size(mu_net, sig2_down ** 0.5, adv_percentile)

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
        '''probs_syn = {label: 0.0 for label in all_labels}
        for i, cluster_idx in enumerate(idx):
            for label, prob in self.outcome_probs.get(str(cluster_idx), {}).items():
                probs_syn[label] += beta[i] * prob
        '''

        # Synthetic analogue-weighted probabilities (safe when kernel-path)
        probs_syn = {label: 0.0 for label in all_labels}
        for i, cluster_idx in enumerate(idx):
            bi = float(beta_arr[i])  # 0.0 if we’re on kernel path
            if bi == 0.0:
                continue
            for label, prob in self.outcome_probs.get(str(int(cluster_idx)), {}).items():
                probs_syn[label] += bi * float(prob)

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
            mu=mu_net,
            sigma=float(var_path),
            variance_down=float(var_down_path),
            residual=float(residual),
            beta=(beta.astype(np.float32) if isinstance(beta, np.ndarray) else None),
            path=final_path,
            k_used=int(idx.size),
            metric=str(self.metric),
            cluster_id=int(idx[0]),
            outcome_probs=blended_probs,
            position_size=float(pos_size),
            drift_ticket=int(ticket_id),
            mu_raw=float(mu_raw),
            p_up=float(p_up),
            p_source=p_source,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha1_list(items: Iterable[str]) -> str:

    feat_str = "|".join(items)
    full = hashlib.sha1(feat_str.encode()).hexdigest()
    return full[:12]
