##############################################
# feature_engineering/pipelines/core.py  (rewritten)
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

from .dataset_loader import load_parquet_dataset
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


class CoreFeaturePipeline:
    """Run feature engineering in‑memory using pure Pandas/SciKit."""

    def __init__(self, parquet_root: str | Path):
        self.parquet_root = Path(parquet_root)
        if not self.parquet_root.exists():
            raise FileNotFoundError(self.parquet_root)

    # ---------------------------------------------------------------------
    @timeit("pipeline‑run")
    def run(
        self,
        symbols: List[str],
        start: _dt.date,
        end: _dt.date,
    ) -> Tuple[pd.DataFrame, dict]:
        """Return features DataFrame + PCA metadata."""

        # 1) Load slice via Arrow Dataset – skip non‑Parquet files implicitly.
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
        for calc in _CALCULATORS:
            logger.debug("Applying %s", calc.__class__.__name__)
            df = calc(df)

        feature_cols = [
            c for c in df.columns if c not in {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
        ]
        features = df[feature_cols].astype(np.float32)

        # 3) Impute → scale → PCA.
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

        # Persist components / scaler params.
        out_dir = self.parquet_root / "_fe_meta"
        out_dir.mkdir(exist_ok=True)
        # ── persist PCA loadings ──────────────────────────────────────────────
        pd.DataFrame(
            pipe.named_steps["pca"].components_.astype(np.float32)
        ).to_parquet(out_dir / "pca_components.parquet", index=False)

        # ── persist scaler parameters (wrap Series in a DataFrame) ───────────
        pd.DataFrame(
            {"scale": pipe.named_steps["scaler"].scale_.astype(np.float32)}
        ).to_parquet(out_dir / "scaler_scale.parquet", index=False)

        pca = pipe.named_steps["pca"]

        pca_meta = {
            "n_components_": int(pca.n_components_),
            "explained_variance_ratio_": pca.explained_variance_ratio_.astype(float).tolist(),
            "components_path": str(out_dir / "pca_components.parquet"),
            "scale_path": str(out_dir / "scaler_scale.parquet"),
        }

        # ------------------------------------------------------------------
        # 4) Assemble final feature table  (raw technicals  +  named PCA comps)
        # ------------------------------------------------------------------

        # a) keep the named technical features we just calculated
        df_named = df[feature_cols].astype(np.float32)

        '''pca_meta = {
                    "n_components_": int(pca.n_components_),
                    "explained_variance_ratio_": pca.explained_variance_ratio_.astype(float).tolist(),
                    "components_path": str(out_dir / "pca_components.parquet"),
                    "scale_path": str(out_dir / "scaler_scale.parquet"),
                }
                feature_df = pd.DataFrame(transformed, index=df.index)
                feature_df["symbol"] = df["symbol"].values
                feature_df["timestamp"] = df["timestamp"].values
                return feature_df, pca_meta'''

        # b) add PCA components with explicit column names
        n_comp = pipe.named_steps["pca"].n_components_
        pca_cols = [f"pca_{i + 1}" for i in range(n_comp)]
        df_pca = pd.DataFrame(
            transformed,
            columns=pca_cols,
            index=df.index,
            dtype=np.float32,
        )

        # c) concatenate and append identifying columns
        feature_df = pd.concat([df_named, df_pca], axis=1)
        feature_df["symbol"] = df["symbol"].values
        feature_df["timestamp"] = df["timestamp"].values
        return feature_df, pca_meta

