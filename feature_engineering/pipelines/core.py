##############################################
# feature_engineering/pipelines/core.py
##############################################
"""Pure‑pandas feature pipeline – now with a concrete ``run`` method."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
]

class CoreFeaturePipeline:
    """Run feature engineering in‑memory using pure Pandas/SciKit."""

    def __init__(self, parquet_root: str | Path):
        self.parquet_root = Path(parquet_root)
        if not self.parquet_root.exists():
            raise FileNotFoundError(self.parquet_root)

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
        # 1) Apply calculators sequentially
        df = df_raw.copy()

        num_cols = ["open", "high", "low", "close", "volume"]
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

        for col in ("Date", "Time"):
            if col in df.columns:
                df.drop(columns=col, inplace=True)

        for calc in _CALCULATORS:
            df = calc(df)

        df = df.ffill().bfill()

        # Select only the features to use for PCA
        features = df[_PREDICT_COLS].astype(np.float32)

        pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=settings.impute_strategy)),
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=settings.pca_variance, svd_solver="full")),
            ]
        )
        transformed = pipe.fit_transform(features)

        # Save PCA/scaler as .npy (consistent with meta)
        out_dir = self.parquet_root / "_fe_meta"
        out_dir.mkdir(exist_ok=True)
        np.save(out_dir / "pca_components.npy", pipe.named_steps["pca"].components_)
        np.save(out_dir / "scaler_scale.npy", pipe.named_steps["scaler"].scale_)

        n_comp = pipe.named_steps["pca"].n_components_
        pca_cols = [f"pca_{i + 1}" for i in range(n_comp)]
        df_pca = pd.DataFrame(transformed, columns=pca_cols, index=df.index, dtype=np.float32)
        df_pca["symbol"] = df["symbol"].values
        df_pca["timestamp"] = df["timestamp"].values

        pca_meta = {
            "n_components": int(n_comp),
            "explained_variance_ratio_": pipe.named_steps["pca"].explained_variance_ratio_.astype(float).tolist(),
            "pca_path": str(out_dir / "pca_components.npy"),
            "scale_path": str(out_dir / "scaler_scale.npy"),
            "predict_cols": _PREDICT_COLS,
        }
        #out["close"] = df["close"].values  # keep close so cluster builder can create y

        return df_pca, pca_meta

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

