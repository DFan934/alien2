##############################################
# feature_engineering/pipelines/core.py
##############################################
"""Pure‑pandas feature pipeline – now with a concrete ``run`` method."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import List, Tuple
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import logging
log = logging.getLogger(__name__)
# AFTER: from typing import List, Tuple
from typing import Literal, Optional

from feature_engineering.pipelines.dataset_loader import load_parquet_dataset, open_parquet_dataset
from feature_engineering.utils import logger, timeit
from feature_engineering.config import settings
from feature_engineering.calculators import (
    VWAPCalculator,
    RVOLCalculator,
    EMA9Calculator,
    EMA20Calculator,
    MomentumCalculator,
    ATRCalculator,
    ADXCalculator,
)

_CALCULATORS = [
    VWAPCalculator(),
    RVOLCalculator(lookback_days=20),
    EMA9Calculator(),
    EMA20Calculator(),
    MomentumCalculator(period=10),
    ATRCalculator(period=14),
    ADXCalculator(period=14),
]

# The only features passed to PCA for dimensionality reduction.
_PREDICT_COLS = [
    "vwap_delta", "rvol_20d", "ema_9_dist", "ema_20_dist", "roc_10",
    "atr_14", "adx_14",
    # trigger-context features:
    #"time_since_trigger_min",
    "time_since_trigger_min",

    "latency_sec",
    "volume_spike_pct",
]

# After _PREDICT_COLS = [...]
FEATURE_ORDER = ["timestamp", "symbol"] + list(_PREDICT_COLS)



# feature_engineering/pipelines/core.py  (near the top, after imports)
def _assert_schema_tz_freq(df: pd.DataFrame) -> None:
    required = {"timestamp","symbol","open","high","low","close","volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"[FE] Missing required columns: {sorted(missing)}")

    # Enforce tz-naive UTC internally: normalize here
    if pd.api.types.is_datetime64tz_dtype(df["timestamp"]):
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)

    # Sort per symbol by timestamp; check monotonicity and duplicates
    df.sort_values(["symbol", "timestamp"], inplace=True)
    dup = df.duplicated(subset=["symbol","timestamp"]).sum()
    if dup:
        # policy: keep last occurrence
        df.drop_duplicates(subset=["symbol","timestamp"], keep="last", inplace=True)

    # Basic OHLC/volume sanity
    if (df[["open","high","low","close","volume"]] < 0).any().any():
        raise ValueError("[FE] Negative OHLC/volume found")

    if not (df["low"] <= df[["open","close"]].min(axis=1)).all():
        raise ValueError("[FE] low must be <= min(open, close)")
    if not (df["high"] >= df[["open","close"]].max(axis=1)).all():
        raise ValueError("[FE] high must be >= max(open, close)")

    # Frequency sanity (approx 1-minute)
    # infer per symbol; warn if median diff not ~60s
    # Frequency sanity & detection: allow 60s or 300s (default), or pick the dominant cadence
    diffs = df.groupby("symbol")["timestamp"].diff().dropna().dt.total_seconds()
    if len(diffs) > 0:
        med = float(diffs.median())
        # Map common cadences to canonical seconds
        candidates = [(60, 45, 75), (300, 270, 330)]
        chosen = None
        for sec, lo, hi in candidates:
            if lo <= med <= hi:
                chosen = sec
                break
        if chosen is None:
            # Try a generic rounding for other clean cadences
            rounded = int(round(med))
            if rounded <= 0:
                raise ValueError(f"[FE] Could not infer a positive cadence (median {med:.1f}s).")
            chosen = rounded
            log.warning("[FE] Non-standard cadence detected (median=%.1fs) → using %ds slots",
                        med, chosen)
        else:
            log.info("[FE] Detected bar cadence: %ds (median=%.1fs)", chosen, med)

        # Propagate cadence to settings for calculators to use
        try:
            from feature_engineering.config import settings
            settings.bar_seconds = int(chosen)
        except Exception:
            # If settings is immutable in your build, stash on DataFrame for local use
            df.attrs["bar_seconds"] = int(chosen)


def _zscore_per_symbol(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Return a copy where `cols` are standardized within each symbol:
    z = (x - mean_symbol) / std_symbol  (std=1 if degenerate).
    """
    out = df.copy()
    def _z(g: pd.DataFrame) -> pd.DataFrame:
        mu = g[cols].mean()
        sig = g[cols].std(ddof=0).replace(0, 1.0)
        g[cols] = (g[cols] - mu) / sig
        return g
    out = out.groupby("symbol", group_keys=False).apply(_z)
    # keep NaNs from rolling calcs consistent
    return out


def _as_utc(s: pd.Series) -> pd.Series:
    """Return a tz-aware UTC datetime64 series from possibly-naive/aware input."""
    s = pd.to_datetime(s, utc=False, errors="coerce")
    # if tz-naive → localize UTC; if tz-aware but not UTC → convert to UTC
    if getattr(s.dt, "tz", None) is None:
        return s.dt.tz_localize("UTC")
    return s.dt.tz_convert("UTC")


class CoreFeaturePipeline:
    """Run feature engineering in‑memory using pure Pandas/SciKit."""

    '''def __init__(self, parquet_root: str | Path):
        self.parquet_root = Path(parquet_root)
        if not self.parquet_root.exists():
            raise FileNotFoundError(self.parquet_root)
    '''
    FEATURE_ORDER = FEATURE_ORDER

    def __init__(self, parquet_root: str | Path):
        self.parquet_root = Path(parquet_root)
        if not self.parquet_root.exists():
            # Allow creation if the intention is to use run_mem, which creates subdirs.
            # raise FileNotFoundError(self.parquet_root)
            pass

        # ADD these lines immediately after self.parquet_root = Path(parquet_root)
        self._pipe: Optional[Pipeline] = None
        self._predict_cols_fitted: Optional[list[str]] = None
        self._normalization_mode_fitted: Optional[str] = None

        # --- NEW: Auto-load the fitted pipeline if it exists ---
        '''pipe_path = self.parquet_root / "_fe_meta" / "pipeline.pkl"
        if pipe_path.exists():
            log.info(f"[FE] Loading pre-fitted pipeline from {pipe_path}")
            self._pipe = joblib.load(pipe_path)
        '''
        # inside CoreFeaturePipeline.__init__
        meta_dir = self.parquet_root / "_fe_meta"
        pipe_path = meta_dir / "pipeline.pkl"
        if pipe_path.exists():
            log.info(f"[FE] Loading pre-fitted pipeline from {pipe_path}")
            self._pipe = joblib.load(pipe_path)

            meta_path = meta_dir / "pca_meta.json"
            self._predict_cols_fitted = None
            if meta_path.exists():
                try:
                    meta_series = pd.read_json(meta_path, typ="series")
                    self._predict_cols_fitted = list(meta_series.get("predict_cols", []))
                    # AFTER: self._predict_cols_fitted = list(meta_series.get("predict_cols", []))
                    self._normalization_mode_fitted = meta_series.get("normalization_mode", "global")
                    log.info("[FE] Loaded fitted normalization_mode=%s", self._normalization_mode_fitted)

                except Exception:
                    log.warning("[FE] Could not read pca_meta.json; will fall back to code _PREDICT_COLS")

        # Hard fallback if meta missing or empty
        '''if not self._predict_cols_fitted:
            self._predict_cols_fitted = list(_PREDICT_COLS)
        if not hasattr(self, "_normalization_mode_fitted") or self._normalization_mode_fitted is None:
            self._normalization_mode_fitted = "global"
        '''
        # REPLACE the two fallback lines with these:
        if not (getattr(self, "_predict_cols_fitted", None)):
            self._predict_cols_fitted = list(_PREDICT_COLS)
        if not (getattr(self, "_normalization_mode_fitted", None)):
            self._normalization_mode_fitted = "global"

    #def fit_mem(self, df_train: pd.DataFrame) -> dict:
    def fit_mem(self, df_train: pd.DataFrame, *,
                    normalization_mode: Literal["global", "per_symbol"] = "global") -> dict:

        """
        Fit imputer+scaler+PCA on TRAIN ONLY. Persist artifacts and return pca_meta.
        """
        df = df_train.copy()
        _assert_schema_tz_freq(df)

        # Phase-2.4: explicitly tolerate missing days per symbol.
        # If a symbol has zero rows in the slice, we simply don't see it here.
        # If a symbol is present but has gaps, calculators will operate on what's there.
        if df.empty:
            raise ValueError("Selected slice returned zero rows – adjust dates/symbols.")

        df_with_features = self._calculate_base_features(df)

        # === NEW: normalization branch ===
        feats_df = df_with_features[_PREDICT_COLS].astype(np.float32)
        if normalization_mode == "per_symbol":
            # standardize within each symbol BEFORE PCA; pipeline will skip global scaler
            df_norm = _zscore_per_symbol(df_with_features, _PREDICT_COLS)
            feats_df = df_norm[_PREDICT_COLS].astype(np.float32)

        X = df_with_features[_PREDICT_COLS].astype(np.float32)

        '''pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy=settings.impute_strategy)),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=settings.pca_variance, svd_solver="full")),
        ])'''

        # === NEW: build pipeline depending on normalization_mode ===
        if normalization_mode == "global":
            pipe = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy=settings.impute_strategy)),
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=settings.pca_variance, svd_solver="full")),
            ])
        else:  # per_symbol
            pipe = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy=settings.impute_strategy)),
                ("pca", PCA(n_components=settings.pca_variance, svd_solver="full")),
            ])

        Xp = pipe.fit_transform(feats_df)

        #Xp = pipe.fit_transform(X)
        log.info("[FE] normalization_mode=%s", normalization_mode)

        out_dir = self.parquet_root / "_fe_meta"
        out_dir.mkdir(parents=True, exist_ok=True)
        #joblib.dump(pipe, out_dir / "pipeline.pkl")
        #joblib.dump(pipe.named_steps["scaler"], out_dir / "scaler.pkl")
        #joblib.dump(pipe.named_steps["pca"], out_dir / "pca.pkl")

        joblib.dump(pipe, out_dir / "pipeline.pkl")
        if "scaler" in pipe.named_steps:
            joblib.dump(pipe.named_steps["scaler"], out_dir / "scaler.pkl")
        joblib.dump(pipe.named_steps["pca"], out_dir / "pca.pkl")


        pca = pipe.named_steps["pca"]
        cum_var = float(pca.explained_variance_ratio_.sum())
        if cum_var + 1e-9 < float(settings.pca_variance):
            raise RuntimeError(f"[FE] PCA retained variance {cum_var:.3%} < target {settings.pca_variance:.3%}")

        pca_meta = {
            "n_components": int(pca.n_components_),
            "explained_variance_ratio_": pca.explained_variance_ratio_.astype(float).tolist(),
            "cum_var": cum_var,
            "predict_cols": list(_PREDICT_COLS),
            "normalization_mode": normalization_mode,  # << NEW
            "created_at": pd.Timestamp.utcnow().isoformat(),
        }
        (out_dir / "pca_meta.json").write_text(pd.Series(pca_meta).to_json(), encoding="utf-8")

        # Keep in-memory handle for immediate transform_mem use
        self._pipe = pipe
        return pca_meta

    # ------------------------------------------------------------------
    # In-memory variant – used by unit-tests & quick back-tests
    # ------------------------------------------------------------------
    def run_mem(self, df_raw: pd.DataFrame, *, normalization_mode: Literal["global","per_symbol"] = "global") -> tuple[pd.DataFrame, dict]:

        """
        Lightweight wrapper around `run()` that skips Arrow/Parquet loading
        and instead receives a *clean* DataFrame (with the same columns
        expected by the calculators).
        Returns ONLY pca_1 … pca_k (+ symbol, timestamp), NOT raw features.
        """

        log.info("[FE] run_mem start – rows=%d", len(df_raw))

        # Ensure trigger_ts exists (scanner may not have set it yet)
        if "trigger_ts" not in df_raw.columns:
            df_raw = df_raw.copy()
            df_raw["trigger_ts"] = df_raw["timestamp"]

        # 1) Apply calculators sequentially
        df = df_raw.copy()
        _assert_schema_tz_freq(df)

        # Phase-2.4: explicitly tolerate missing days per symbol.
        # If a symbol has zero rows in the slice, we simply don't see it here.
        # If a symbol is present but has gaps, calculators will operate on what's there.
        if df.empty:
            raise ValueError("Selected slice returned zero rows – adjust dates/symbols.")



        num_cols = ["open", "high", "low", "close", "volume"]
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

        # Drop stray cols if present
        for col in ("Date", "Time"):
            if col in df.columns:
                df.drop(columns=col, inplace=True)

        for calc in _CALCULATORS:
            t0 = pd.Timestamp.utcnow()
            df = calc(df)
            log.debug("[FE] %-14s rows=%d  dt=%.0f ms",
                      calc.__class__.__name__,
                      len(df),
                      (pd.Timestamp.utcnow() - t0).total_seconds() * 1e3)

        df = df.ffill().bfill()

        # ─── compute trigger context features ──────────────────────────────
        # assumes snapshot had 'trigger_ts' and 'volume_spike_pct'
        #df["time_since_trigger_min"] = (
        #    df["timestamp"] - df["trigger_ts"]
        #).dt.total_seconds().div(60)

        # REPLACE the raw subtraction with a guarded delta in seconds:
        '''delta_sec = (
            (df["timestamp"] - df["trigger_ts"]).dt.total_seconds().astype("float32")
            if "trigger_ts" in df.columns
            else pd.Series(0.0, index=df.index, dtype="float32")
        )'''
        # REPLACE the raw subtraction with tz-aligned version
        if "trigger_ts" not in df.columns:
            df["trigger_ts"] = df["timestamp"]

        ts_utc = _as_utc(df["timestamp"])
        tr_utc = _as_utc(df["trigger_ts"])
        delta_sec = (ts_utc - tr_utc).dt.total_seconds().astype("float32")
        #df["latency_sec"] = delta_sec

        # Example: if you were previously doing
        # df["latency_sec"] = (df["timestamp"] - df["trigger_ts"]).dt.total_seconds().astype("float32")
        # then do:
        #df["latency_sec"] = delta_sec

        #df["volume_spike_pct"] = df["volume_spike_pct"]

        df_with_features = self._calculate_base_features(df_raw)

        '''
        # Select only the features to use for PCA
        #features = df[_PREDICT_COLS].astype(np.float32)
        features = df_with_features[_PREDICT_COLS].astype(np.float32)


        pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=settings.impute_strategy)),
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=settings.pca_variance, svd_solver="full")),
            ]
        )
        transformed = pipe.fit_transform(features)
        '''

        # === NEW: normalization branch ===
        feats_df = df_with_features[_PREDICT_COLS].astype(np.float32)
        if normalization_mode == "per_symbol":
            df_norm = _zscore_per_symbol(df_with_features, _PREDICT_COLS)
            feats_df = df_norm[_PREDICT_COLS].astype(np.float32)

        # === NEW: build pipeline depending on normalization_mode ===
        if normalization_mode == "global":
            pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy=settings.impute_strategy)),
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=settings.pca_variance, svd_solver="full")),
                ]
            )
        else:
            pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy=settings.impute_strategy)),
                    ("pca", PCA(n_components=settings.pca_variance, svd_solver="full")),
                ]
            )

        transformed = pipe.fit_transform(feats_df)


        self._pipe = pipe

        pca = pipe.named_steps["pca"]

        cum_var = float(pca.explained_variance_ratio_.sum())
        target = float(settings.pca_variance)  # e.g., 0.95
        if cum_var + 1e-9 < target:
            raise RuntimeError(
                f"[FE] PCA retained variance {cum_var:.3%} < target {target:.3%} — adjust _PREDICT_COLS or pca_variance.")


        log.info("[FE] PCA n_comp=%d  cum_var=%.1f%%",
                 pca.n_components_,
                 pca.explained_variance_ratio_.sum() * 100)




        # Save PCA/scaler as .npy (consistent with meta)
        out_dir = self.parquet_root / "_fe_meta"
        out_dir.mkdir(exist_ok=True)
        #np.save(out_dir / "pca_components.npy", pipe.named_steps["pca"].components_)
        #np.save(out_dir / "scaler_scale.npy", pipe.named_steps["scaler"].scale_)

        np.save(out_dir / "pca_components.npy", pipe.named_steps["pca"].components_)
        if "scaler" in pipe.named_steps:
            np.save(out_dir / "scaler_scale.npy", pipe.named_steps["scaler"].scale_)


        # ALSO persist the actual fitted objects for eval-time transform (A2)
        #joblib.dump(pipe.named_steps["scaler"], out_dir / "scaler.pkl")
        #joblib.dump(pipe.named_steps["pca"], out_dir / "pca.pkl")

        if "scaler" in pipe.named_steps:
            joblib.dump(pipe.named_steps["scaler"], out_dir / "scaler.pkl")
        joblib.dump(pipe.named_steps["pca"], out_dir / "pca.pkl")


        joblib.dump(pipe, out_dir / "pipeline.pkl")


        n_comp = pipe.named_steps["pca"].n_components_
        pca_cols = [f"pca_{i + 1}" for i in range(n_comp)]
        #df_pca = pd.DataFrame(transformed, columns=pca_cols, index=df.index, dtype=np.float32)
        df_pca = pd.DataFrame(transformed, columns=pca_cols, index=df_with_features.index, dtype=np.float32)

        #df_pca["symbol"] = df["symbol"].values
        #df_pca["timestamp"] = df["timestamp"].values

        df_pca["symbol"] = df_with_features["symbol"].values
        df_pca["timestamp"] = df_with_features["timestamp"].values
        log.info("[FE] normalization_mode=%s", normalization_mode)

        pca_meta = {
            "n_components": int(n_comp),
            "explained_variance_ratio_": pipe.named_steps["pca"].explained_variance_ratio_.astype(float).tolist(),
            "pca_path": str(out_dir / "pca_components.npy"),
            "scale_path": str(out_dir / "scaler_scale.npy"),
            "predict_cols": _PREDICT_COLS,
            "normalization_mode": normalization_mode,  # << NEW

        }

        # Persist meta so transform-only paths can recover predict_cols & normalization_mode
        try:
            (out_dir / "pca_meta.json").write_text(pd.Series({
                **pca_meta,
                "created_at": pd.Timestamp.utcnow().isoformat(),
            }).to_json(), encoding="utf-8")
        except Exception as e:
            log.warning("[FE] Could not persist pca_meta.json: %s", e)

        #out["close"] = df["close"].values  # keep close so cluster builder can create y

        return df_pca, pca_meta

        # In feature_engineering/pipelines/core.py, inside the CoreFeaturePipeline class

    def _calculate_base_features(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Applies all base feature calculators and adds context features."""
        # If assert stashed cadence on attrs (immutable settings), mirror it into settings now
        if "bar_seconds" in df_raw.attrs:
            try:
                from feature_engineering.config import settings
                settings.bar_seconds = int(df_raw.attrs["bar_seconds"])
            except Exception:
                pass



        df = df_raw.copy()

        # Ensure numeric types before calculations
        num_cols = ["open", "high", "low", "close", "volume"]
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

        if "Date" in df.columns and "Time" in df.columns:
            df = df.drop(columns=["Date", "Time"])

        for calc in _CALCULATORS:
            df = calc(df)

        # This forward/backward fill is crucial for stability
        df = df.ffill().bfill()

        # ---- Trigger-context features (robust defaults when scanner data absent) ----
        '''if "trigger_ts" in df.columns:
            df["time_since_trigger_min"] = (
                    df["timestamp"] - df["trigger_ts"]
            ).dt.total_seconds().div(60)
        else:
            df["time_since_trigger_min"] = 0.0
        '''

        if "trigger_ts" not in df.columns:
            df["trigger_ts"] = df["timestamp"]

        ts_utc = _as_utc(df["timestamp"])
        tr_utc = _as_utc(df["trigger_ts"])
        delta_sec = (ts_utc - tr_utc).dt.total_seconds().astype("float32")
        df["latency_sec"] = delta_sec

        df["time_since_trigger_min"] = (delta_sec / 60.0).astype("float32")


        if "volume_spike_pct" not in df.columns:
            df["volume_spike_pct"] = 0.0

        # VWAP delta: session-reset
        if "vwap_delta" not in df.columns:
            vwap_calc = VWAPCalculator(name="vwap_delta")
            vwap_df = vwap_calc.transform(df)  # uses session_id internally
            df = pd.concat([df, vwap_df], axis=1)

        # RVOL: minute-of-session baseline across previous N sessions
        if "rvol_20d" not in df.columns:
            rvol_calc = RVOLCalculator(days=20, name="rvol_20d")
            rvol_df = rvol_calc.transform(df)
            df = pd.concat([df, rvol_df], axis=1)

        return df

    @classmethod
    def from_artifact_dir(cls, artefact_dir: Path) -> "CoreFeaturePipeline":
        """
        Reload the StandardScaler and PCA that were saved by
        PathClusterEngine.build().
        """
        artefact_dir = Path(artefact_dir)
        scaler = joblib.load(artefact_dir / "scaler.pkl")
        pca = joblib.load(artefact_dir / "pca.pkl")
        inst = cls(parquet_root=Path(""))  # dummy root
        inst._scaler = scaler  # type: ignore[attr-defined]
        inst._pca = pca  # type: ignore[attr-defined]
        return inst

    # -----------------------------------------------------------------------
    # 🔄 Transform in‑memory DataFrame without re‑fitting
    # -----------------------------------------------------------------------
    '''def transform_mem(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the saved scaler+PCA _without_ refitting.
        """
        #X_raw = df[self.raw_feature_cols].to_numpy(dtype=np.float32)

        # Use the same predictor list as run/run_mem
        predict_cols = [
            "vwap_delta", "rvol_20d", "ema_9_dist", "ema_20_dist", "roc_10",
            "atr_14", "adx_14", "time_since_trigger_min", "volume_spike_pct"
        ]
        X_raw = df[predict_cols].to_numpy(dtype=np.float32)

        X_scaled = self._scaler.transform(X_raw)  # type: ignore[attr-defined]
        X_pca = self._pca.transform(X_scaled)  # type: ignore[attr-defined]
        feats = pd.DataFrame(
            X_pca,
            columns=[f"pca_{i + 1}" for i in range(X_pca.shape[1])],
            index=df.index,
        )
        feats["symbol"] = df["symbol"].values
        feats["timestamp"] = df["timestamp"].values
        return feats'''

    # In feature_engineering/pipelines/core.py

    def transform_mem(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates all features, then TRANSFORMS using the pre-fitted pipeline.
        """


        #if not hasattr(self, '_pipe'):
        #    raise RuntimeError("Pipeline has not been fitted yet. Call run_mem() first.")

        # at the top of transform_mem, right after the docstring:
        if not hasattr(self, "_pipe"):
            raise RuntimeError("[FE] No fitted pipeline found. Call fit_mem() or provide artefacts.")
        df = df_raw.copy()
        _assert_schema_tz_freq(df)

        # Phase-2.4: explicitly tolerate missing days per symbol.
        # If a symbol has zero rows in the slice, we simply don't see it here.
        # If a symbol is present but has gaps, calculators will operate on what's there.
        if df.empty:
            raise ValueError("Selected slice returned zero rows – adjust dates/symbols.")

        # AFTER features/raw copy and BEFORE any use of 'trigger_ts'
        if "trigger_ts" not in df_raw.columns:
            df_raw = df_raw.copy()
            df_raw["trigger_ts"] = df_raw["timestamp"]

        # 1) Apply all base feature calculations
        df_with_features = self._calculate_base_features(df_raw)

        # --- Back-compat & schema alignment guard ---------------------------------
        # Ensure both names exist so we can satisfy either schema
        '''if "latency_sec" in df_with_features and "time_since_trigger_min" not in df_with_features:
            df_with_features["time_since_trigger_min"] = (
                    df_with_features["latency_sec"] / 60.0
            ).astype("float32")

        if "time_since_trigger_min" in df_with_features and "latency_sec" not in df_with_features:
            df_with_features["latency_sec"] = (
                    df_with_features["time_since_trigger_min"] * 60.0
            ).astype("float32")
        '''
        # in transform_mem (after computing base features)
        if "latency_sec" in df_with_features and "time_since_trigger_min" not in df_with_features:
            df_with_features["time_since_trigger_min"] = (df_with_features["latency_sec"] / 60.0).astype("float32")
        if "time_since_trigger_min" in df_with_features and "latency_sec" not in df_with_features:
            df_with_features["latency_sec"] = (df_with_features["time_since_trigger_min"] * 60.0).astype("float32")

        # Prefer the feature order the fitted pipeline was actually trained on
        expected = list(getattr(self._pipe, "feature_names_in_", [])) \
                   or list(getattr(self, "_predict_cols_fitted", [])) \
                   or list(_PREDICT_COLS)  # final fallback only

        # Add any missing expected cols with neutral defaults
        for c in expected:
            if c not in df_with_features:
                df_with_features[c] = 0.0

        # Drop extras and enforce dtype/order to match training
        features = df_with_features[expected].astype(np.float32)
        # --------------------------------------------------------------------------

        # then:
        transformed = self._pipe.transform(features)

        # AFTER: df_with_features = self._calculate_base_features(df_raw)

        # === NEW: respect the fitted normalization mode ===
        #predict_cols = getattr(self, "_predict_cols_fitted", _PREDICT_COLS)
        #predict_cols = getattr(self, "_predict_cols_fitted", None) or _PREDICT_COLS

        # after df_with_features = self._calculate_base_features(df_raw)
        predict_cols = getattr(self, "_predict_cols_fitted", None) or _PREDICT_COLS

        # Back-compat guard: if minutes col is requested, synthesize it from latency_sec
        if "time_since_trigger_min" in predict_cols and "time_since_trigger_min" not in df_with_features.columns:
            if "latency_sec" in df_with_features.columns:
                df_with_features["time_since_trigger_min"] = (
                        df_with_features["latency_sec"] / 60.0
                ).astype("float32")

        features_df = df_with_features[predict_cols].astype(np.float32)

        missing = [c for c in predict_cols if c not in df_with_features.columns]
        if missing:
            # Try code defaults once more (covers old metas)
            fallback = _PREDICT_COLS
            missing_fb = [c for c in fallback if c not in df_with_features.columns]
            if not missing_fb:
                predict_cols = fallback
            else:
                raise RuntimeError(
                    "[FE] Transform columns missing from dataset.\n"
                    f"  requested: {predict_cols}\n"
                    f"  missing:   {missing}\n"
                    f"  available: {sorted(df_with_features.columns.tolist())[:60]}..."
                )

        norm_mode = getattr(self, "_normalization_mode_fitted", "global")
        if norm_mode == "per_symbol":
            df_norm = _zscore_per_symbol(df_with_features, list(predict_cols))
            features_df = df_norm[predict_cols].astype(np.float32)

        # Use features_df for transform in the expected order
        features = features_df

        # 2) Select only the features to use for PCA
        #features = df_with_features[_PREDICT_COLS].astype(np.float32)

        # 3) Transform using the fitted pipeline
        transformed = self._pipe.transform(features)

        n_comp = self._pipe.named_steps["pca"].n_components_
        pca_cols = [f"pca_{i + 1}" for i in range(n_comp)]
        df_pca = pd.DataFrame(transformed, columns=pca_cols, index=df_with_features.index, dtype=np.float32)
        df_pca["symbol"] = df_with_features["symbol"].values
        df_pca["timestamp"] = df_with_features["timestamp"].values

        return df_pca

    # ---------------------------------------------------------------------
    @timeit("pipeline‑run")
    def run(
            self,
            symbols: List[str],
            start: _dt.date,
            end: _dt.date,
            *,
            normalization_mode: Literal["global", "per_symbol"] = "global",
    ) -> Tuple[pd.DataFrame, dict]:
        """Return features DataFrame + PCA metadata.
        Outputs ONLY pca_1 … pca_k (+ symbol, timestamp), NOT raw features.
        """

        logger.info("Loading Parquet slice …")
        try:
            dataset = open_parquet_dataset(self.parquet_root)

        except (FileNotFoundError, pa.ArrowInvalid) as exc:
            raise RuntimeError(f"Failed to open dataset: {exc}") from exc

        # Build a filter expression (>= / <= for older PyArrow).
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        date_filter = (
            (ds.field("timestamp") >= start_ts) &
            (ds.field("timestamp") <= end_ts) &
            (ds.field("symbol").isin(symbols))
        )
        arrow_table = dataset.to_table(filter=date_filter)

        if arrow_table.num_rows == 0:
            raise ValueError("Selected slice returned zero rows – adjust dates/symbols.")

        df = arrow_table.to_pandas()
        _assert_schema_tz_freq(df)

        # Phase-2.4: explicitly tolerate missing days per symbol.
        # If a symbol has zero rows in the slice, we simply don't see it here.
        # If a symbol is present but has gaps, calculators will operate on what's there.
        if df.empty:
            raise ValueError("Selected slice returned zero rows – adjust dates/symbols.")

        logger.info("Loaded %d rows, columns: %s", len(df), list(df.columns))

        # Phase-2.4: per-symbol coverage snapshot for this slice
        per_sym_counts = df.groupby("symbol", as_index=False)["timestamp"].count().rename(columns={"timestamp": "rows"})
        try:
            # Compact "sym:rows" line for quick eyeballing
            snap = ", ".join([f"{r.symbol}:{int(r.rows)}" for r in per_sym_counts.itertuples()])
            logger.info("[Slice] rows by symbol → %s", snap)
        except Exception:
            pass

        # 2) Apply calculators sequentially.
        '''logger.info("After load:        %d rows", len(df))
        for calc in _CALCULATORS:
            df = calc(df)
            logger.info("After %-12s %d rows", calc.__class__.__name__, len(df))
        df = df.ffill().bfill()

        # ─── compute trigger context features ──────────────────────────────
        df["time_since_trigger_min"] = (
            df["timestamp"] - df["trigger_ts"]
        ).dt.total_seconds().div(60)
        df["volume_spike_pct"] = df["volume_spike_pct"]

        logger.info("After ffill/bfill: %d rows", len(df))
        '''

        # AFTER
        logger.info("After load:        %d rows", len(df))
        df = self._calculate_base_features(df)
        logger.info("After base feats:  %d rows", len(df))

        # === NEW: normalization branch ===
        feats_df = df[_PREDICT_COLS].astype(np.float32)
        if normalization_mode == "per_symbol":
            df_norm = _zscore_per_symbol(df, _PREDICT_COLS)
            feats_df = df_norm[_PREDICT_COLS].astype(np.float32)

        # === NEW: build pipeline depending on normalization_mode ===
        if normalization_mode == "global":
            pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy=settings.impute_strategy)),
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=settings.pca_variance, svd_solver="full")),
                ]
            )
        else:
            pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy=settings.impute_strategy)),
                    ("pca", PCA(n_components=settings.pca_variance, svd_solver="full")),
                ]
            )

        transformed = pipe.fit_transform(feats_df)
        log.info("[FE] normalization_mode=%s", normalization_mode)

        _assert_schema_tz_freq(df)

        # Phase-2.4: explicitly tolerate missing days per symbol.
        # If a symbol has zero rows in the slice, we simply don't see it here.
        # If a symbol is present but has gaps, calculators will operate on what's there.
        if df.empty:
            raise ValueError("Selected slice returned zero rows – adjust dates/symbols.")

        # If a pre-fitted pipeline exists (loaded in __init__), do a transform-only run to avoid leakage.
        if hasattr(self, "_pipe") and self._pipe is not None:
            logger.info("[FE] Transform-only path (fitted pipeline detected)")
            df_features = self._calculate_base_features(df)

            #X = df_features[_PREDICT_COLS].astype(np.float32)
            #Xp = self._pipe.transform(X)

            '''predict_cols = getattr(self, "_predict_cols_fitted", _PREDICT_COLS)

            missing = set(predict_cols) - set(df_features.columns)
            if missing:
                raise RuntimeError(f"[FE] Transform columns missing from dataset: {sorted(missing)}. "
                                   f"Either refit pipeline or align feature builders.")
            '''

            # >>> NEW: derive expected order from the fitted pipeline
            try:
                expected = list(self._pipe.named_steps["imputer"].feature_names_in_)
            except Exception:
                # Fallback if imputer not present or no names stored
                expected = list(getattr(self._pipe, "feature_names_in_", []))

            if not expected:
                raise RuntimeError(
                    "[FE] Loaded pipeline has no feature_names_in_; "
                    "refit pipeline or ensure it was trained on a pandas DataFrame."
                )

            missing = [c for c in expected if c not in df_features.columns]
            extra = [c for c in df_features.columns if
                     c in expected and c not in expected]  # unused, but here for clarity
            if missing:
                raise RuntimeError(
                    f"[FE] Transform columns missing from dataset: {missing}. "
                    "Either refit the pipeline or realign feature builders/_PREDICT_COLS."
                )

            #X = df_features[predict_cols].astype(np.float32)  # critical: use saved order
            # Use the exact order expected by the pipeline
            X = df_features.loc[:, expected].astype(np.float32)

            Xp = self._pipe.transform(X)

            # Build output with pca_* + base cols we keep
            pca = self._pipe.named_steps["pca"]
            pcols = [f"pca_{i:02d}" for i in range(pca.n_components_)]
            out = pd.DataFrame(Xp, columns=pcols, index=df_features.index)
            keep = [c for c in ["symbol", "timestamp", "close"] if c in df_features.columns]
            out[keep] = df_features[keep]
            pca_meta = {
                "n_components": int(pca.n_components_),
                "explained_variance_ratio_": pca.explained_variance_ratio_.astype(float).tolist(),
                "cum_var": float(pca.explained_variance_ratio_.sum()),
                "predict_cols": list(_PREDICT_COLS),
                "created_at": pd.Timestamp.utcnow().isoformat(),
                "normalization_mode": normalization_mode,
            }
            return out.reset_index(drop=True), pca_meta

        features = df[_PREDICT_COLS].astype(np.float32)

        pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=settings.impute_strategy)),
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=settings.pca_variance, svd_solver="full")),
            ]
        )

        transformed = pipe.fit_transform(features)
        logger.info(
            "PCA kept %d components (%.1f %% var)",
            pipe.named_steps["pca"].n_components_,
            settings.pca_variance * 100,
        )

        out_dir = self.parquet_root / "_fe_meta"
        out_dir.mkdir(exist_ok=True)
        #np.save(out_dir / "pca_components.npy", pipe.named_steps["pca"].components_)
        #np.save(out_dir / "scaler_scale.npy", pipe.named_steps["scaler"].scale_)

        # ALSO persist the actual fitted objects for eval-time transform (A2)
        #joblib.dump(pipe.named_steps["scaler"], out_dir / "scaler.pkl")
        #joblib.dump(pipe.named_steps["pca"], out_dir / "pca.pkl")


        np.save(out_dir / "pca_components.npy", pipe.named_steps["pca"].components_)
        if "scaler" in pipe.named_steps:
            np.save(out_dir / "scaler_scale.npy", pipe.named_steps["scaler"].scale_)

        if "scaler" in pipe.named_steps:
            joblib.dump(pipe.named_steps["scaler"], out_dir / "scaler.pkl")
        joblib.dump(pipe.named_steps["pca"], out_dir / "pca.pkl")


        # ADD THIS LINE to save the whole pipeline object
        joblib.dump(pipe, out_dir / "pipeline.pkl")

        k = pipe.named_steps["pca"].n_components_
        pca_cols = [f"pca_{i + 1}" for i in range(k)]
        out = pd.DataFrame(transformed, columns=pca_cols, index=df.index, dtype=np.float32)
        out["symbol"] = df["symbol"].values
        out["timestamp"] = df["timestamp"].values

        pca_meta = {
            "n_components": int(k),
            "explained_variance_ratio_": pipe.named_steps["pca"].explained_variance_ratio_.astype(float).tolist(),
            "pca_path": str(out_dir / "pca_components.npy"),
            "scale_path": str(out_dir / "scaler_scale.npy"),
            "predict_cols": _PREDICT_COLS,
        }
        out["close"] = df["close"].values  # keep close so cluster builder can create y

        # after assembling pca_meta dict in run_mem:
        (out_dir / "pca_meta.json").write_text(pd.Series(pca_meta).to_json(), encoding="utf-8")

        return out, pca_meta

    # === NEW: slice helpers to avoid leakage ===
    def fit_slice(self, symbols, start, end):
        """
        Load TRAIN slice from Parquet, calculate base features, then fit & persist pipeline.
        """
        table = ds.dataset(str(self.parquet_root), format="parquet")
        filt = (
                (ds.field("symbol").isin(symbols)) &
                (ds.field("timestamp") >= pd.Timestamp(start)) &
                (ds.field("timestamp") < pd.Timestamp(end))
        )
        arrow = table.to_table(filter=filt)
        df = arrow.to_pandas(types_mapper=pd.ArrowDtype)
        return self.fit_mem(df)

    def transform_slice(self, symbols, start, end):
        """
        Load EVAL slice from Parquet and transform with pre-fitted pipeline; never fits.
        """
        if not hasattr(self, "_pipe"):
            # Try auto-load if present
            meta = (self.parquet_root / "_fe_meta" / "pipeline.pkl")
            if meta.exists():
                self._pipe = joblib.load(meta)
            else:
                raise RuntimeError("[FE] No fitted pipeline; call fit_slice or fit_mem first.")

        table = ds.dataset(str(self.parquet_root), format="parquet")
        filt = (
                (ds.field("symbol").isin(symbols)) &
                (ds.field("timestamp") >= pd.Timestamp(start)) &
                (ds.field("timestamp") < pd.Timestamp(end))
        )
        arrow = table.to_table(filter=filt)
        df = arrow.to_pandas(types_mapper=pd.ArrowDtype)
        _assert_schema_tz_freq(df)

        # Phase-2.4: explicitly tolerate missing days per symbol.
        # If a symbol has zero rows in the slice, we simply don't see it here.
        # If a symbol is present but has gaps, calculators will operate on what's there.
        if df.empty:
            raise ValueError("Selected slice returned zero rows – adjust dates/symbols.")

        df_feat = self._calculate_base_features(df)
        X = df_feat[_PREDICT_COLS].astype(np.float32)

        # Transform ONLY
        Xp = self._pipe.transform(X)

        # Assemble output: pca_* + base identifiers
        pca = self._pipe.named_steps["pca"]
        cols = [f"pca_{i}" for i in range(pca.n_components_)]
        out = pd.DataFrame(Xp, columns=cols, index=df_feat.index)
        out = pd.concat([df_feat[["symbol", "timestamp"]].reset_index(drop=True), out.reset_index(drop=True)], axis=1)
        return out

    # ---------------------------------------------------------------------------
    # 💡 NEW: factory that loads the *frozen* preprocessing from an artefact dir
    # ---------------------------------------------------------------------------





