# ──────────────────────────────────────────────────────────────────────────
# pipelines/dask_pipeline.py  (completed)
# ──────────────────────────────────────────────────────────────────────────
"""Dask‑based scalable feature pipeline.

Uses the same public interface as :class:`feature_engineering.pipelines.core.FeaturePipeline`.

Example
-------
>>> from feature_engineering.pipelines.dask_pipeline import DaskFeaturePipeline
>>> pipe = DaskFeaturePipeline()
>>> df, pca_meta = pipe.run(
...     parquet_root="parquet",
...     symbols=["AAPL", "MSFT"],
...     start="2019-01-01",
...     end="2019-12-31",
...     npartitions=4,
... )
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple, List

import dask.dataframe as dd
import pandas as pd
from dask.diagnostics import ProgressBar
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA

from feature_engineering.utils import logger
from feature_engineering.calculators import (
    vwap as _vwap,
    rvol as _rvol,
    ema as _ema,
    momentum as _mom,
    atr as _atr,
    adx as _adx,
)
from feature_engineering.patterns import candlestick as _candle
from feature_engineering.reducers.pca import PCAReducer

__all__ = ["DaskFeaturePipeline"]


class DaskFeaturePipeline:
    """Drop‑in replacement for the pandas FeaturePipeline but using Dask."""

    def __init__(self, npartitions: int | None = None):
        self.npartitions = npartitions or 8  # sensible default on 8‑core boxes

        # --- calculators ordered list --------------------------------------------------
        self._calculators = [
            _vwap.VWAPCalculator(),
            _rvol.RVOLCalculator(window=20),
            _ema.EMACalculator(span=9),
            _ema.EMACalculator(span=20),
            _mom.MomentumCalculator(window=10),
            _atr.ATRCalculator(window=14),
            _adx.ADXCalculator(window=14),
            _candle.CandlestickCalculator(),
        ]

    # ---------------------------------------------------------------------
    # Public API – mirrors core.FeaturePipeline
    # ---------------------------------------------------------------------
    def run(
        self,
        parquet_root: str | Path,
        symbols: Iterable[str],
        start: str,
        end: str,
        *,
        explained_var: float = 0.95,
        impute_thresh: float = 0.2,
    ) -> Tuple[pd.DataFrame, dict]:
        """Load data ➜ compute features ➜ fit PCA ➜ return pandas DF + meta."""

        ds = dd.read_parquet(
            str(parquet_root),
            engine="pyarrow",
            filters=[("symbol", "in", list(symbols))],
            chunksize="16MB",
        )
        # Date filter – convert to pandas Timestamp for comparison
        ds = ds[(ds.timestamp >= pd.Timestamp(start, tz="UTC")) & (ds.timestamp <= pd.Timestamp(end, tz="UTC"))]

        logger.info("Dask read: %s rows across %d partitions", ds.shape[0].compute(), ds.npartitions)

        # Ensure proper partitioning
        if self.npartitions:
            ds = ds.repartition(npartitions=self.npartitions)

        # Sequentially apply each calculator via map_partitions
        for calc in self._calculators:
            logger.info("Applying %s…", calc.name)
            ds = ds.map_partitions(calc.transform, meta=calc.meta())

        # Drop rows with too many NaNs, then impute the rest (mean per column)
        missing_frac = ds.isna().mean(axis=1)
        ds = ds[missing_frac <= impute_thresh]
        ds = ds.map_partitions(lambda d: d.fillna(d.mean()), meta=ds._meta)

        # Convert to pandas for PCA (IncrementalPCA processes batches)
        with ProgressBar():
            pdf = ds.compute()

        feature_cols = pdf.columns.drop(["timestamp", "symbol"])
        scaler = StandardScaler()
        ipca = IncrementalPCA(n_components=explained_var, whiten=False)
        pca_pipe: Pipeline = Pipeline([("scaler", scaler), ("pca", ipca)])
        features_reduced = pca_pipe.fit_transform(pdf[feature_cols])

        pca_meta = {
            "explained_variance_ratio_": ipca.explained_variance_ratio_.tolist(),
            "components_": ipca.components_.tolist(),
            "n_components_": ipca.n_components_.item() if hasattr(ipca.n_components_, "item") else int(ipca.n_components_),
        }

        final_df = pd.concat(
            [pdf[["timestamp", "symbol"]].reset_index(drop=True), pd.DataFrame(features_reduced)],
            axis=1,
        )
        return final_df, pca_meta