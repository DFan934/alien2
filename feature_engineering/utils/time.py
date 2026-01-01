# feature_engineering/utils/time.py
from __future__ import annotations

from typing import Any, Literal, Optional
import pandas as pd


FloorUnit = Literal["min", "s", "ms", "us", "ns"]


def to_utc(ts: Any) -> Any:
    """
    Canonical timestamp coercion:
    - Accepts Series / Index / arraylike / scalar
    - Returns tz-aware UTC timestamps (never tz-naive)
    """
    if ts is None:
        return ts

    # Series
    if isinstance(ts, pd.Series):
        out = pd.to_datetime(ts, utc=True, errors="coerce")
        return out

    # Index
    if isinstance(ts, pd.Index):
        # to_datetime(Index) returns Index; enforce utc
        out = pd.to_datetime(ts, utc=True, errors="coerce")
        return out

    # Scalar or arraylike
    return pd.to_datetime(ts, utc=True, errors="coerce")


def to_utc_floor(ts: Any, unit: FloorUnit = "min") -> Any:
    """tz-aware UTC + flooring."""
    out = to_utc(ts)
    if isinstance(out, pd.Series):
        return out.dt.floor(unit)
    if isinstance(out, pd.DatetimeIndex):
        return out.floor(unit)
    # scalar Timestamp
    try:
        return out.floor(unit)  # type: ignore[attr-defined]
    except Exception:
        return out


def ensure_utc_timestamp_col(df: pd.DataFrame, col: str = "timestamp", *, who: str = "") -> None:
    """
    In-place: enforce df[col] is tz-aware UTC datetime64[ns, UTC].
    Raises if coercion creates NaT.
    """
    if col not in df.columns:
        raise ValueError(f"{who} missing required column: {col}")

    before_dtype = str(df[col].dtype)
    df[col] = to_utc(df[col])

    nat = int(df[col].isna().sum())
    if nat:
        raise ValueError(f"{who} {col} coercion produced NaT={nat}")

    after_dtype = str(df[col].dtype)
    if after_dtype != "datetime64[ns, UTC]":
        raise ValueError(f"{who} {col} dtype not UTC-aware: before={before_dtype} after={after_dtype}")
