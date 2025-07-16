# ──────────────────────────────────────────────────────────────────────────
# run_pipeline.py – “one‑click” feature build for the ENTIRE dataset
# ──────────────────────────────────────────────────────────────────────────
"""Generate engineered features for **all** symbols and all dates.

Simply run:

    python -m feature_engineering.run_pipeline          # pandas
    python -m feature_engineering.run_pipeline --engine dask --npartitions 8

You may still override paths or symbol/date subsets with flags, but nothing
is required.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import pyarrow.dataset as ds
import pyarrow as pa
from scanner.schema import FEATURE_ORDER
from feature_engineering.output_writer import write_feature_dataset
from feature_engineering.pipelines.core import CoreFeaturePipeline
from feature_engineering.pipelines.dask_pipeline import DaskFeaturePipeline
from feature_engineering.utils import logger

# ---------------------------------------------------------------------------
# Project-root helpers
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]        # <project>/
INPUT_PARQUET_ROOT  = PROJECT_ROOT / "parquet"            #  ⟵ raw minute bars
OUTPUT_PARQUET_ROOT = PROJECT_ROOT / "feature_engineering" / "feature_parquet"

# ---------------------------------------------------------------------------
# Discover symbols by scanning the *input* dataset
# ---------------------------------------------------------------------------
def _discover_symbols(input_root: Path) -> List[str]:
    symbols = [p.name.split("=")[-1]
               for p in input_root.rglob("symbol=*") if p.is_dir()]
    if not symbols:
        raise FileNotFoundError(f"No symbol= folders under {input_root}")
    return sorted(set(symbols))

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full-dataset feature pipeline")
    p.add_argument("--input_root",  default=str(INPUT_PARQUET_ROOT),
                   help="Input minute-bar Parquet root (from data_ingestion)")
    p.add_argument("--output_root", default=str(OUTPUT_PARQUET_ROOT),
                   help="Output Parquet root for engineered features")
    p.add_argument("--symbols", nargs="*", help="Tickers to include (default: all)")
    p.add_argument("--start", help="Start date YYYY-MM-DD (default: min)")
    p.add_argument("--end",   help="End   date YYYY-MM-DD (default: max)")
    p.add_argument("--engine", choices=["pandas", "dask"],
                   default="pandas", help="Execution backend")
    p.add_argument("--npartitions", type=int, default=None,
                   help="Dask partitions (only if --engine dask)")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args         = _parse_args()
    input_root   = Path(args.input_root).expanduser().resolve()
    output_root  = Path(args.output_root).expanduser().resolve()

    symbols = args.symbols or _discover_symbols(input_root)
    logger.info("Discovered %d symbols", len(symbols))

    # Choose backend
    if args.engine == "dask":
        pipe = DaskFeaturePipeline(parquet_root=input_root,
                                   npartitions=args.npartitions)
    else:
        pipe = CoreFeaturePipeline(parquet_root=input_root)

    df, pca_meta = pipe.run(
        symbols=symbols,
        start=args.start or "1900-01-01",
        end=args.end   or "2100-01-01",
    )

    # Replace any unnamed numeric columns with pca_X labels so schema hash is stable
    unnamed = [c for c in df.select_dtypes("number").columns if isinstance(c, int) or c == ""]
    for i, col in enumerate(unnamed, 1):
        df.rename(columns={col: f"feat_{i}"}, inplace=True)

    # Reorder dataframe columns to canonical FEATURE_ORDER before writing
    df = df[list(FEATURE_ORDER)]


    write_feature_dataset(df, output_root)
    logger.info("Saved features → %s", output_root)
    logger.info("PCA retained %d components covering %.2f%% variance",
                pca_meta.get("n_components", pca_meta.get("n_components_")),
                sum(pca_meta["explained_variance_ratio_"]) * 100)

if __name__ == "__main__":      # pragma: no cover
    main()