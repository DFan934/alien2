# ========================
# file: data_ingestion/persistence.py
# ========================
"""Lightweight Parquet + state helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Parquet helpers
# ------------------------------------------------------------------

def write_parquet(df: pd.DataFrame, path: Path, partition_cols: list[str]):
    df = df.copy()
    if "year" in partition_cols and "year" not in df.columns:
        if "timestamp" in df.columns:
            df["year"] = df["timestamp"].dt.year
        else:
            raise KeyError("'year' partition requested but no timestamp column")


    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df)
    pq.write_to_dataset(table, root_path=str(path), partition_cols=partition_cols)
    logger.info("Wrote %s rows → %s", len(df), path)


def read_parquet(folder: Path) -> pd.DataFrame:
    return pq.read_table(str(folder)).to_pandas()


# ------------------------------------------------------------------
# Simple JSON state (circuit‑breaker, etc.)
# ------------------------------------------------------------------

def save_state(path: Path | str, state: Mapping[str, Any]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf‑8") as fh:
        json.dump(state, fh)


def load_state(path: Path | str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf‑8"))


