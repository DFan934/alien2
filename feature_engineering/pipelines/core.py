##############################################
# feature_engineering/pipelines/core.py  (rewritten)
##############################################
"""Pure‑pandas feature pipeline – now with a concrete ``run`` method."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd
import pyarrow.dataset as ds, pathlib
#print(ds.dataset(pathlib.Path("parquet")).schema)     # adjust path as needed

import pyarrow as pa       # keep this at top with the other imports

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_engineering.calculators import ALL_CALCULATORS
from feature_engineering.utils import logger, timeit

# ---------------------------------------------------------------------------
# Helper: build default sklearn pipeline (impute → scale → PCA 95 %)
# ---------------------------------------------------------------------------

def _build_reducer(variance: float = 0.95) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=variance, svd_solver="full")),
        ]
    )

import pyarrow as pa, pandas as pd

import numpy as np, pyarrow as pa, pandas as pd

def _arrow_ts(value: str) -> pa.Scalar:
    """
    Arrow scalar matching Parquet column type timestamp[ns] (no tz).
    Works regardless of the precision Pandas chooses for the input string.
    """
    ns_int = pd.Timestamp(value).value           # int64 nanoseconds since epoch
    ts_ns  = np.datetime64(ns_int, 'ns')         # numpy.datetime64[ns]
    return pa.scalar(ts_ns, type=pa.timestamp('ns'))
  # exact match
# ---------------------------------------------------------------------------
# Core pipeline class
# ---------------------------------------------------------------------------

class CoreFeaturePipeline:
    """End‑to‑end orchestrator (pandas path)."""

    def __init__(
        self,
        parquet_root: str | Path,
        calculators: Sequence = ALL_CALCULATORS,
        *,
        pca_variance: float = 0.95,
    ) -> None:
        self.parquet_root = Path(parquet_root).expanduser().resolve()
        self.calculators = list(calculators)
        self.reducer = _build_reducer(pca_variance)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @timeit("CoreFeaturePipeline.run")
    def run(
        self,
        *,
        symbols: List[str],
        start: str,
        end: str,
    ) -> Tuple[pd.DataFrame, dict]:
        """Return (features_df, pca_metadata)."""
        logger.info("Loading Parquet slice …")
        scanner = (
            ds.dataset(self.parquet_root, partitioning="hive")
            .scanner(
                filter=(
                        ds.field("symbol").isin(symbols)
                        & (ds.field("timestamp") >= _arrow_ts(start))
                        & (ds.field("timestamp") <= _arrow_ts(end))
                ),
                columns=["timestamp", "open", "high", "low", "close", "volume", "symbol"],
            )
        )

        df = scanner.to_table().to_pandas()
        if df.empty:
            raise ValueError("No rows returned for requested slice")

        # --- core indicators ------------------------------------------------
        for calc in self.calculators:
            df = pd.concat([df, calc.transform(df)], axis=1)

        # --- dimensionality reduction --------------------------------------
        feature_cols = [c for c in df.columns if c not in {"timestamp", "symbol"}]
        numeric_block = df[feature_cols]
        reduced = self.reducer.fit_transform(numeric_block)
        pca_cols = [f"pc{i}" for i in range(1, reduced.shape[1] + 1)]
        df[pca_cols] = reduced

        meta = {
            "n_components_": self.reducer.named_steps["pca"].n_components_,
            "explained_variance_ratio_": self.reducer.named_steps["pca"].explained_variance_ratio_.tolist(),
        }
        return df, meta
