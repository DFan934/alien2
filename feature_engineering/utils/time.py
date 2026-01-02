# feature_engineering/utils/time.py
from __future__ import annotations

from typing import Any, Literal, Optional
import pandas as pd


FloorUnit = Literal["min", "s", "ms", "us", "ns"]


'''def to_utc(ts: Any) -> Any:
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
'''

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype

def to_utc(x):
    """
    Convert timestamps to tz-aware UTC datetimes.

    - tz-aware datetimes/strings -> UTC
    - naive datetimes/strings    -> assume UTC and localize
    - preserves NaN/None as NaT
    """
    # Series / Index / array-like
    if isinstance(x, (pd.Series, pd.Index)):
        s = x

        # Already datetime64[ns, tz]
        if is_datetime64tz_dtype(s.dtype):
            out = s.tz_convert("UTC") if isinstance(s, pd.DatetimeIndex) else s.dt.tz_convert("UTC")
            return out

        # Already datetime64[ns] (naive)
        if is_datetime64_any_dtype(s.dtype):
            out = s.tz_localize("UTC") if isinstance(s, pd.DatetimeIndex) else s.dt.tz_localize("UTC")
            return out

        # Object / strings (possibly mixed tz + naive)
        out = pd.to_datetime(s, utc=True, errors="coerce")

        # Fallback pass for any remaining parse failures (common when mixed formats exist)
        mask = out.isna() & s.notna()
        if mask.any():
            reparsed = pd.to_datetime(s[mask], errors="coerce")

            # reparsed may be tz-aware or naive; normalize to UTC
            if is_datetime64tz_dtype(reparsed.dtype):
                out.loc[mask] = reparsed.dt.tz_convert("UTC")
            else:
                out.loc[mask] = reparsed.dt.tz_localize("UTC")

        return out

    # Scalar
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return pd.NaT

    ts = pd.to_datetime(x, utc=True, errors="coerce")
    if pd.isna(ts):
        ts2 = pd.to_datetime(x, errors="coerce")
        if pd.isna(ts2):
            return pd.NaT
        # ts2 may be naive; treat as UTC
        if getattr(ts2, "tzinfo", None) is None:
            return ts2.tz_localize("UTC")
        return ts2.tz_convert("UTC")
    return ts


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
