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

from feature_engineering.pipelines.dataset_loader import load_parquet_dataset
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
    "time_since_trigger_min",
    "volume_spike_pct",
]

class CoreFeaturePipeline:
    """Run feature engineering in‑memory using pure Pandas/SciKit."""

    '''def __init__(self, parquet_root: str | Path):
        self.parquet_root = Path(parquet_root)
        if not self.parquet_root.exists():
            raise FileNotFoundError(self.parquet_root)
    '''

    def __init__(self, parquet_root: str | Path):
        self.parquet_root = Path(parquet_root)
        if not self.parquet_root.exists():
            # Allow creation if the intention is to use run_mem, which creates subdirs.
            # raise FileNotFoundError(self.parquet_root)
            pass

        # --- NEW: Auto-load the fitted pipeline if it exists ---
        pipe_path = self.parquet_root / "_fe_meta" / "pipeline.pkl"
        if pipe_path.exists():
            log.info(f"[FE] Loading pre-fitted pipeline from {pipe_path}")
            self._pipe = joblib.load(pipe_path)


    # ------------------------------------------------------------------
    # In-memory variant – used by unit-tests & quick back-tests
    # ------------------------------------------------------------------
    def run_mem(
            self,
            df_raw: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Lightweight wrapper around `run()` that skips Arrow/Parquet loading
        and instead receives a *clean* DataFrame (with the same columns
        expected by the calculators).
        Returns ONLY pca_1 … pca_k (+ symbol, timestamp), NOT raw features.
        """

        log.info("[FE] run_mem start – rows=%d", len(df_raw))

        # 1) Apply calculators sequentially
        df = df_raw.copy()

        num_cols = ["open", "high", "low", "close", "volume"]
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

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
        df["time_since_trigger_min"] = (
            df["timestamp"] - df["trigger_ts"]
        ).dt.total_seconds().div(60)
        df["volume_spike_pct"] = df["volume_spike_pct"]

        df_with_features = self._calculate_base_features(df_raw)


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

        self._pipe = pipe

        pca = pipe.named_steps["pca"]
        log.info("[FE] PCA n_comp=%d  cum_var=%.1f%%",
                 pca.n_components_,
                 pca.explained_variance_ratio_.sum() * 100)

        # Save PCA/scaler as .npy (consistent with meta)
        out_dir = self.parquet_root / "_fe_meta"
        out_dir.mkdir(exist_ok=True)
        np.save(out_dir / "pca_components.npy", pipe.named_steps["pca"].components_)
        np.save(out_dir / "scaler_scale.npy", pipe.named_steps["scaler"].scale_)

        # ALSO persist the actual fitted objects for eval-time transform (A2)
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

        pca_meta = {
            "n_components": int(n_comp),
            "explained_variance_ratio_": pipe.named_steps["pca"].explained_variance_ratio_.astype(float).tolist(),
            "pca_path": str(out_dir / "pca_components.npy"),
            "scale_path": str(out_dir / "scaler_scale.npy"),
            "predict_cols": _PREDICT_COLS,
        }
        #out["close"] = df["close"].values  # keep close so cluster builder can create y

        return df_pca, pca_meta

        # In feature_engineering/pipelines/core.py, inside the CoreFeaturePipeline class

    def _calculate_base_features(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Applies all base feature calculators and adds context features."""
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

        # Compute trigger-context features if the columns exist
        if "trigger_ts" in df.columns:
            df["time_since_trigger_min"] = (
                    df["timestamp"] - df["trigger_ts"]
            ).dt.total_seconds().div(60)

        # The column should already exist from run_backtest.py, just ensure it
        if "volume_spike_pct" not in df.columns:
            df["volume_spike_pct"] = 0.0

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
        if not hasattr(self, '_pipe'):
            raise RuntimeError("Pipeline has not been fitted yet. Call run_mem() first.")

        # 1) Apply all base feature calculations
        df_with_features = self._calculate_base_features(df_raw)

        # 2) Select only the features to use for PCA
        features = df_with_features[_PREDICT_COLS].astype(np.float32)

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
    ) -> Tuple[pd.DataFrame, dict]:
        """Return features DataFrame + PCA metadata.
        Outputs ONLY pca_1 … pca_k (+ symbol, timestamp), NOT raw features.
        """

        logger.info("Loading Parquet slice …")
        try:
            dataset = load_parquet_dataset(self.parquet_root)
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
        logger.info("Loaded %d rows, columns: %s", len(df), list(df.columns))

        # 2) Apply calculators sequentially.
        logger.info("After load:        %d rows", len(df))
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
        np.save(out_dir / "pca_components.npy", pipe.named_steps["pca"].components_)
        np.save(out_dir / "scaler_scale.npy", pipe.named_steps["scaler"].scale_)

        # ALSO persist the actual fitted objects for eval-time transform (A2)
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

        return out, pca_meta

    # ---------------------------------------------------------------------------
    # 💡 NEW: factory that loads the *frozen* preprocessing from an artefact dir
    # ---------------------------------------------------------------------------





