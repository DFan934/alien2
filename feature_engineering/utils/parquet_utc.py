# feature_engineering/utils/parquet_utc.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from feature_engineering.utils.time import ensure_utc_timestamp_col


@dataclass(frozen=True)
class TimestampDtypeError(RuntimeError):
    path: str
    col: str
    dtype: str

    def __str__(self) -> str:
        return f"[UTC] TimestampDtypeError: {self.path} col={self.col} dtype={self.dtype} (expected datetime64[ns, UTC])"


def _as_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def write_parquet_utc(
    df: pd.DataFrame,
    path: str | Path,
    *,
    timestamp_cols: Iterable[str] = ("timestamp", "decision_ts", "entry_ts", "exit_ts"),
    compression: str = "snappy",
) -> None:
    """
    HARD INVARIANT:
      Any timestamp column that exists must be tz-aware UTC before writing.
    """
    p = _as_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # enforce in-place on a shallow copy to avoid surprising callers
    out = df.copy()
    for c in timestamp_cols:
        if c in out.columns:
            ensure_utc_timestamp_col(out, c, who=f"[WRITE {p.name}]")

    out.to_parquet(p, index=False, compression=compression)


def read_parquet_utc(
    path: str | Path,
    *,
    timestamp_cols: Iterable[str] = ("timestamp", "decision_ts", "entry_ts", "exit_ts"),
    strict: bool = True,
) -> pd.DataFrame:
    """
    HARD INVARIANT:
      Any timestamp column that exists must round-trip as tz-aware UTC.
    If strict=True, tz-naive datetime64[ns] is a hard failure (signals a broken writer).
    """
    p = _as_path(path)
    df = pd.read_parquet(p)

    for c in timestamp_cols:
        if c not in df.columns:
            continue

        dtype_str = str(df[c].dtype)

        # If the parquet came back tz-naive, we refuse (this is the core bug Task 6 is preventing).
        if strict and dtype_str == "datetime64[ns]":
            raise TimestampDtypeError(str(p), c, dtype_str)

        # Otherwise, coerce and validate (covers object/string cases, legacy CSV pipelines, etc.)
        ensure_utc_timestamp_col(df, c, who=f"[READ {p.name}]")

    return df
