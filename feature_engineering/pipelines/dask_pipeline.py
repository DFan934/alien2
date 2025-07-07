# ============================================================================
# feature_engineering/pipelines/dask_pipeline.py
# ============================================================================
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Tuple

import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
from dask_ml.impute import SimpleImputer
from dask_ml.preprocessing import StandardScaler
from dask_ml.decomposition import PCA

from feature_engineering.calculators import (
    VWAPCalculator,
    RVOLCalculator,
    EMA9Calculator,
    EMA20Calculator,
    MomentumCalculator,
    ATRCalculator,
    ADXCalculator,
)
from feature_engineering.config import settings
from feature_engineering.pipelines.dataset_loader import load_parquet_dataset
from feature_engineering.utils import logger, timeit

__all__ = ["DaskFeaturePipeline"]

_CALCULATORS = [
    VWAPCalculator(),
    RVOLCalculator(lookback_days=20),
    EMA9Calculator(),
    EMA20Calculator(),
    MomentumCalculator(period=10),
    ATRCalculator(period=14),
    ADXCalculator(period=14),
]


class DaskFeaturePipeline:
    """Dask‑based feature builder – parallel + lazy."""

    def __init__(self, parquet_root: Path | str, npartitions: int | None = None):
        self.parquet_root = Path(parquet_root)
        self.npartitions = npartitions

    # ------------------------------------------------------------------
    @timeit("dask‑run")
    def run(
        self,
        symbols: Iterable[str],
        start: str | datetime,
        end: str | datetime,
    ) -> Tuple[pd.DataFrame, dict]:
        dataset = load_parquet_dataset(self.parquet_root)
        filt = dataset.field("symbol").isin(list(symbols))
        df = dataset.to_table(filter=filt).to_pandas()

        df = df[(df["timestamp"] >= pd.Timestamp(start)) & (df["timestamp"] <= pd.Timestamp(end))]
        if df.empty:
            raise FileNotFoundError("No rows match symbol/date slice – aborting.")

        ddf = dd.from_pandas(df, npartitions=self.npartitions or 4)

        # Apply calculators sequentially
        for calc in _CALCULATORS:
            ddf = calc(ddf)

        numeric = list(ddf.select_dtypes("number").columns)

        pipe = PCA(n_components=settings.pca_variance, svd_solver="full")
        ddf[numeric] = (
            StandardScaler().fit_transform(
                SimpleImputer(strategy=settings.impute_strategy).fit_transform(ddf[numeric])
            ).map_blocks(pipe.fit_transform)
        )
        df_final = ddf.compute()

        pca_meta = {
            "n_components_": pipe.n_components_,
            "explained_variance_ratio_": pipe.explained_variance_ratio_.tolist(),
        }
        return df_final, pca_meta
