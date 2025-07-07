# ===========================================================================
# feature_engineering/pipelines/dataset_loader.py  (NEW OR REPLACED)
# ---------------------------------------------------------------------------
"""Robust Parquet dataset loader that works on any PyArrow version."""
from __future__ import annotations

from pathlib import Path
from typing import List

import pyarrow as pa
import pyarrow.dataset as ds

__all__ = ["load_parquet_dataset"]


def _file_list(root: Path) -> List[Path]:
    files = list(root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {root}")
    return files


def load_parquet_dataset(root: Path) -> ds.Dataset:  # noqa: D401
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