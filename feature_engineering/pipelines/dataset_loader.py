# ===========================================================================
# feature_engineering/pipelines/dataset_loader.py  (NEW OR REPLACED)
# ---------------------------------------------------------------------------
"""Robust Parquet dataset loader that works on any PyArrow version."""
from __future__ import annotations

from pathlib import Path
from typing import List

import pyarrow as pa
import pyarrow.dataset as ds


# AFTER: `import pyarrow.dataset as ds`
from typing import Sequence, Tuple, Optional
import numpy as np
import pandas as pd
import pyarrow.compute as pc


# AFTER: __all__ = ["load_parquet_dataset"]
__all__ = ["open_parquet_dataset", "load_parquet_dataset", "load_slice"]


def _file_list(root: Path) -> List[Path]:
    files = list(root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {root}")
    return files


'''def open_parquet_dataset(root: Path) -> ds.Dataset:  # noqa: D401
    """Return an Arrow *Dataset* compatible with any PyArrow version.

    * PyArrow ≥ 14: use the fast path with ``ignore_invalid_files`` so artifacts
      like ``schema.json`` are skipped automatically.
    * Older PyArrow: fall back to an explicit ``*.parquet`` file list.
    """
    try:
        return ds.dataset(
            str(root),
            format="parquet",
            partitioning="hive",
            ignore_invalid_files=True,
        )
    except TypeError:  # keyword not supported → old PyArrow
        return ds.dataset(_file_list(root), format="parquet", partitioning="hive")
    ds.dataset(
            str(root),
            format="parquet",
            partitioning="hive",
            ignore_invalid_files=True,  # type: ignore[arg-type]
        )

    files = [p for p in root.rglob("*.parquet")]
    if not files:
        raise FileNotFoundError(f"No parquet files under {root}")
    return ds.dataset(files, format="parquet", partitioning="hive")
'''

def open_parquet_dataset(root: Path) -> ds.Dataset:
    try:
        return ds.dataset(
            str(root),
            format="parquet",
            partitioning="hive",
            ignore_invalid_files=True,
        )
    except TypeError:
        files = list(root.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files under {root}")
        return ds.dataset(files, format="parquet", partitioning="hive")


# === New helpers and public API for multi-symbol loading ===

def _normalize_time_bounds(dataset: ds.Dataset, start, end) -> Tuple[pa.Scalar, pa.Scalar]:
    # Find timestamp field/type
    try:
        ts_field = next(f for f in dataset.schema if f.name == "timestamp")
    except StopIteration:
        raise KeyError("Dataset schema is missing a `timestamp` column")
    ts_type = ts_field.type  # pa.timestamp('ns', tz='UTC') or pa.timestamp('ns')

    # Coerce to pandas Timestamps
    start_ts = pd.Timestamp(start)
    end_ts   = pd.Timestamp(end)

    if getattr(ts_type, "tz", None):  # tz-aware Arrow field
        start_ts = start_ts.tz_convert("UTC") if start_ts.tzinfo else start_ts.tz_localize("UTC")
        end_ts   = end_ts.tz_convert("UTC")   if end_ts.tzinfo   else end_ts.tz_localize("UTC")
    else:  # tz-naive Arrow field
        if start_ts.tzinfo:
            start_ts = start_ts.tz_convert("UTC").tz_localize(None)
        if end_ts.tzinfo:
            end_ts = end_ts.tz_convert("UTC").tz_localize(None)

    # Build tz-correct Arrow scalars (matches ns + tz)
    start_s = pa.scalar(start_ts.to_pydatetime(), type=ts_type)
    end_s   = pa.scalar(end_ts.to_pydatetime(),   type=ts_type)
    return start_s, end_s



def _project_columns(
    dataset: ds.Dataset,
    wanted: Optional[Sequence[str]],
) -> Sequence[str]:
    """
    Decide the projection column list. If `wanted` is None, use minimal FE columns.
    """
    schema_names = {f.name for f in dataset.schema}
    if wanted is None:
        wanted = ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
    missing = [c for c in wanted if c not in schema_names]
    if missing:
        raise KeyError(f"Projection columns not found in dataset: {missing}")
    return list(wanted)


def load_parquet_dataset(
    root: "str | Path",
    symbols: Sequence[str],
    start: "pd.Timestamp | str",
    end: "pd.Timestamp | str",
    *,
    columns: Optional[Sequence[str]] = None,
    batch_size: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load and stack rows for `symbols` within [start, end] using a single Arrow filter.
    Preserves hive partition pruning (symbol=/year=/month=), returns a pandas DataFrame.

    Parameters
    ----------
    root : str | Path
        Hive-partitioned parquet root (symbol=/year=/month=).
    symbols : list[str]
        Universe of tickers to load.
    start, end : pd.Timestamp | str
        Inclusive time window bounds. Accepts tz-aware or naive; will be coerced to dataset type.
    columns : list[str] | None
        Projection; defaults to minimal FE columns.
    batch_size : int | None
        Optional scanner batch size.

    Returns
    -------
    pd.DataFrame with at least: ['timestamp','symbol','open','high','low','close','volume', ...]
    """
    root = Path(root)
    dataset = open_parquet_dataset(root)

    # Validate symbol field presence once
    if "symbol" not in {f.name for f in dataset.schema}:
        raise KeyError("Dataset schema is missing a `symbol` column")

    # Prepare filter scalars to match dataset timestamp type
    start_s, end_s = _normalize_time_bounds(dataset, start, end)

    # Build a single composite filter
    filt = (
        (ds.field("timestamp") >= start_s)
        & (ds.field("timestamp") <= end_s)
        & (ds.field("symbol").isin(list(symbols)))
    )

    # Columns/projection
    cols = _project_columns(dataset, columns)

    # Execute scan → Arrow table (PyArrow-version compatible)
    # In load_parquet_dataset, replace the Scanner block with:

    # Execute scan → Arrow table (PyArrow-version compatible)
    try:
        scan_kwargs = {
            "filter": filt,
            "columns": cols,
        }
        if batch_size is not None:
            scan_kwargs["batch_size"] = int(batch_size)

        scanner = ds.Scanner.from_dataset(dataset, **scan_kwargs)
        table = scanner.to_table()
    except AttributeError:
        # Older APIs without Scanner/scan
        table = dataset.to_table(filter=filt, columns=cols)

    # Execute scan → Arrow table
    #scanner = dataset.scan(filter=filt, columns=cols, batch_size=batch_size)
    #table = scanner.to_table()

    # Convert to pandas; then enforce dtypes post-hoc as needed
    df = table.to_pandas(types_mapper=None)  # keep Arrow->pandas defaults

    # Canonicalize dtypes for FE (quant criteria expect these)
    # Timestamps: ensure UTC tz-aware pandas index/column
    if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        # If naive, make it UTC; if tz-aware but not UTC, convert to UTC
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

    # Prices as float32; volume as int32; symbol as str (object)
    for c in ("open", "high", "low", "close"):
        if c in df.columns:
            df[c] = df[c].astype("float32", copy=False)
    if "volume" in df.columns:
        # Guard against nulls before casting to int32
        if df["volume"].isna().any():
            df["volume"] = df["volume"].fillna(0)
        df["volume"] = df["volume"].astype("int32", copy=False)
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype("string").astype("object")  # pandas string→object

    # Stable ordering for downstream grouping/joins
    if "symbol" in df.columns:
        df = df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    else:
        df = df.sort_values(["timestamp"]).reset_index(drop=True)

    return df


def load_slice(
    root: "str | Path",
    symbols: Sequence[str],
    start: "pd.Timestamp | str",
    end: "pd.Timestamp | str",
    *,
    columns: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
    """
    Thin wrapper: returns (stacked_df, minute_clock_index) for later joins.

    For Phase 1, the clock is simply the *union* of timestamps present in the slice.
    A canonical trading clock (market hours/holidays) will be introduced in later phases.
    """
    df = load_parquet_dataset(root, symbols, start, end, columns=columns)

    if df.empty:
        clock = pd.DatetimeIndex([], tz="UTC", name="timestamp")
        return df, clock

    # Build a simple union clock at bar frequency implied by data (assumes uniform minute bars)
    # We use unique timestamps across all symbols to avoid look-ahead; later we’ll promote to canonical.
    clock = pd.DatetimeIndex(df["timestamp"].unique()).tz_convert("UTC")
    clock = clock.sort_values()
    clock.name = "timestamp"

    return df, clock



