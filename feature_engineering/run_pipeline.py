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


# AFTER: from feature_engineering.utils import logger
from typing import Tuple
import pyarrow.dataset as ds
from feature_engineering.pipelines.dataset_loader import load_slice
from feature_engineering.pipelines.core import CoreFeaturePipeline  # ensure direct import is available

# ---- Step-3: mode toggle ----
# "CLI" → obey CLI args
# "NOCLI" → ignore CLI args and run *all* symbols for the *entire* time range present
PIPELINE_MODE: str = "NOCLI"  # change to "NOCLI" for full-dataset/no-CLI mode


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

# AFTER: def _discover_symbols(input_root: Path) -> List[str]:
def _discover_time_bounds(input_root: Path) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Scan a hive dataset and return (min_ts, max_ts) over all fragments."""
    dataset = ds.dataset(str(input_root), format="parquet", partitioning="hive", exclude_invalid_files=True)
    # Use a filtered projection to avoid pulling all columns
    # If stats are available, Arrow will prune efficiently; otherwise, this is still acceptable for Step-3.
    tcol = "timestamp"
    scanner = dataset.scanner(columns=[tcol])
    tbl = scanner.to_table()
    s = pd.to_datetime(tbl.column(tcol).to_pandas(), utc=True)
    return s.min(), s.max()


def _partition_counts(input_root: Path, symbols: List[str]) -> str:
    """For logging: return 'sym:n, sym:n, ...' of year/month partitions per symbol."""
    parts = []
    for s in symbols:
        p = input_root / f"symbol={s}"
        n = len(list(p.glob("year=*/month=*")))
        parts.append((s, n))
    parts.sort(key=lambda x: x[0])
    return ", ".join([f"{s}:{n}" for s, n in parts])


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

    # AFTER: p.add_argument("--npartitions", type=int, default=None,
    #                      help="Dask partitions (only if --engine dask)")
    p.add_argument("--normalization", choices=["global", "per_symbol"], default="global",
                   help="Feature normalization mode (Step-2): global or per_symbol")

    return p.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args         = _parse_args()
    input_root   = Path(args.input_root).expanduser().resolve()
    output_root  = Path(args.output_root).expanduser().resolve()

    # AFTER: input_root = ..., output_root = ...
    if PIPELINE_MODE.upper() == "NOCLI":
        # Run **all symbols** and **entire time range**
        symbols = _discover_symbols(input_root)
        start_ts, end_ts = _discover_time_bounds(input_root)
        start_str, end_str = start_ts.isoformat(), end_ts.isoformat()
        logger.info("[Phase-3] NOCLI mode: %d symbols, %s → %s", len(symbols), start_str, end_str)
        normalization = "per_symbol"  # default you can change here if you want
    else:
        symbols = args.symbols or _discover_symbols(input_root)
        start_str = args.start or "1900-01-01"
        end_str = args.end or "2100-01-01"
        normalization = args.normalization

    symbols = args.symbols or _discover_symbols(input_root)
    logger.info("Discovered %d symbols", len(symbols))
    logger.info("[Universe] %d symbols", len(symbols))
    logger.info("[Window]   %s → %s", start_str, end_str)
    logger.info("[Parquet]   Partitions: %s", _partition_counts(input_root, symbols))

    # Choose backend
    if args.engine == "dask":
        pipe = DaskFeaturePipeline(parquet_root=input_root,
                                   npartitions=args.npartitions)
    else:
        pipe = CoreFeaturePipeline(parquet_root=input_root)

    '''df, pca_meta = pipe.run(
        symbols=symbols,
        start=args.start or "1900-01-01",
        end=args.end   or "2100-01-01",
    )'''

    if args.engine == "dask":
        # Keep Dask path as-is for now (Step-3 focuses on pandas+loader).
        df, pca_meta = pipe.run(symbols=symbols, start=start_str, end=end_str,
                                normalization_mode=normalization)
    else:
        # Step-3: explicit read via Arrow loader, then in-memory run
        df_raw, clock = load_slice(input_root, symbols, start_str, end_str)
        logger.info("[Loader] rows=%d, unique_timestamps=%d", len(df_raw), len(clock))
        pipe = CoreFeaturePipeline(parquet_root=input_root)
        df, pca_meta = pipe.run_mem(df_raw, normalization_mode=normalization)

    # Simple throughput baseline (rows/sec) for Step-3 acceptance
    rows = len(df)
    if rows:
        logger.info("[FE] Output rows: %d", rows)

    # Replace any unnamed numeric columns with pca_X labels so schema hash is stable
    unnamed = [c for c in df.select_dtypes("number").columns if isinstance(c, int) or c == ""]
    for i, col in enumerate(unnamed, 1):
        df.rename(columns={col: f"feat_{i}"}, inplace=True)

    # Reorder dataframe columns to canonical FEATURE_ORDER before writing
    #df = df[list(FEATURE_ORDER)]

    # With this safer logic:
    pca_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("pca_")]
    base_cols = [c for c in ["symbol", "timestamp", "close"] if c in df.columns]
    other_cols = [c for c in df.columns if c not in pca_cols + base_cols]

    if 'FEATURE_ORDER' in globals() and set(FEATURE_ORDER).issubset(df.columns):
        # If the schema declares an order that's actually present, use it.
        df = df[list(FEATURE_ORDER)]
    else:
        # Otherwise, write PCA-first, then base, then anything extra (stable + future-proof)
        df = df[pca_cols + base_cols + other_cols]

    write_feature_dataset(df, output_root)
    logger.info("Saved features → %s", output_root)
    logger.info("PCA retained %d components covering %.2f%% variance",
                pca_meta.get("n_components", pca_meta.get("n_components_")),
                sum(pca_meta["explained_variance_ratio_"]) * 100)

if __name__ == "__main__":      # pragma: no cover
    main()