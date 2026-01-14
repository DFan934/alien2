from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from feature_engineering.utils.time import ensure_utc_timestamp_col


@dataclass(frozen=True)
class GridAuditRow:
    symbol: str
    expected_freq_s: int
    min_ts: str
    max_ts: str
    n_rows_in: int
    n_rows_out: int
    duplicates_dropped: int
    median_delta_s_in: float | None
    median_delta_s_out: float | None
    missing_ratio_out: float
    tz_dtype_ok: bool
    seconds_on_boundary_ratio: float


class GridAuditError(RuntimeError):
    pass


# -----------------------------
# Phase 3: Unified clock (observed minutes only) + enforcement
# -----------------------------

import hashlib
from dataclasses import asdict
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd


ClockPolicy = Literal["union_observed", "min_symbols_observed"]


class ClockMismatchError(RuntimeError):
    """Raised when a downstream stage uses timestamps not on the run's unified clock."""


def compute_clock_hash(clock_index: pd.DatetimeIndex) -> str:
    """
    Deterministic hash of the unified clock.
    Uses int64 ns timestamps (UTC) to avoid timezone string differences.
    """
    if clock_index is None or len(clock_index) == 0:
        return "empty"

    idx = pd.DatetimeIndex(clock_index)
    if idx.tz is None:
        # normalize to UTC
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")

    payload = idx.view("int64").tobytes()
    return hashlib.sha1(payload).hexdigest()


def build_unified_clock(
    bars: pd.DataFrame,
    *,
    policy: ClockPolicy = "union_observed",
    min_symbols: int = 1,
    ts_col: str = "timestamp",
    symbol_col: str = "symbol",
    presence_col: str = "bar_present",
    freq: str = "60s",
) -> Tuple[pd.DatetimeIndex, Dict[str, object]]:
    """
    Build the *observed minutes only* unified clock.

    policy:
      - union_observed: include any minute with >=1 symbol present
      - min_symbols_observed: include minutes with >=min_symbols present

    presence definition:
      - if presence_col exists: present := numeric(presence_col) > 0
      - else if 'close' exists: present := close.notna()
      - else: present := True (last resort; discouraged)
    """
    if bars is None or len(bars) == 0:
        return pd.DatetimeIndex([], tz="UTC"), {
            "policy": policy,
            "min_symbols": int(min_symbols),
            "n_minutes": 0,
            "ts_min": None,
            "ts_max": None,
            "presence_basis": "empty",
        }

    b = bars.copy()

    # timestamps -> UTC, floored to grid
    b[ts_col] = pd.to_datetime(b[ts_col], utc=True, errors="coerce")
    b = b.dropna(subset=[ts_col, symbol_col])
    b[ts_col] = b[ts_col].dt.floor(freq)

    # present mask
    presence_basis = None
    if presence_col in b.columns:
        bp = pd.to_numeric(b[presence_col], errors="coerce").fillna(0)
        b["_present"] = (bp > 0)
        presence_basis = presence_col
    elif "close" in b.columns:
        b["_present"] = b["close"].notna()
        presence_basis = "close.notna"
    else:
        b["_present"] = True
        presence_basis = "fallback_true"

    # reduce to observed rows
    b = b[b["_present"]]

    if len(b) == 0:
        return pd.DatetimeIndex([], tz="UTC"), {
            "policy": policy,
            "min_symbols": int(min_symbols),
            "n_minutes": 0,
            "ts_min": None,
            "ts_max": None,
            "presence_basis": presence_basis,
        }

    counts = b.groupby(ts_col)[symbol_col].nunique()

    if policy == "union_observed":
        keep = counts.index
    elif policy == "min_symbols_observed":
        keep = counts.index[counts >= int(min_symbols)]
    else:
        raise ValueError(f"Unknown clock policy: {policy!r}")

    clock = pd.DatetimeIndex(pd.to_datetime(keep, utc=True)).sort_values()

    meta = {
        "policy": policy,
        "min_symbols": int(min_symbols),
        "n_minutes": int(len(clock)),
        "ts_min": str(clock.min()) if len(clock) else None,
        "ts_max": str(clock.max()) if len(clock) else None,
        "presence_basis": presence_basis,
        "n_unique_ts_present": int(len(counts)),
        "n_ts_meeting_threshold": int(len(clock)),
    }
    return clock, meta


def assert_df_on_clock(
    df: pd.DataFrame,
    *,
    clock_index: pd.DatetimeIndex,
    expected_clock_hash: str,
    ts_col: str = "timestamp",
    who: str = "df",
) -> None:
    """
    Hard enforcement: all timestamps used by df must be subset of clock_index.
    """
    if clock_index is None or len(clock_index) == 0:
        raise ClockMismatchError(f"[{who}] unified_clock is empty; cannot validate timestamps.")

    if df is None or len(df) == 0:
        return

    if ts_col not in df.columns:
        raise ClockMismatchError(f"[{who}] missing ts_col={ts_col!r}; cannot validate clock usage.")

    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    bad_nat = int(ts.isna().sum())
    if bad_nat:
        raise ClockMismatchError(f"[{who}] has {bad_nat} NaT timestamps; cannot validate clock usage.")

    # subset check
    clock_set = set(pd.DatetimeIndex(clock_index).tz_convert("UTC"))
    uniq = pd.DatetimeIndex(ts.unique()).tz_convert("UTC")
    extra = [t for t in uniq if t not in clock_set]
    if extra:
        extra_sorted = sorted(extra)[:10]
        raise ClockMismatchError(
            f"[{who}] timestamps not in unified_clock (showing up to 10): {extra_sorted}"
        )

    # hash consistency check (cheap sanity)
    actual_hash = compute_clock_hash(clock_index)
    if actual_hash != expected_clock_hash:
        raise ClockMismatchError(
            f"[{who}] clock hash mismatch: expected={expected_clock_hash} actual={actual_hash}"
        )



def _median_delta_seconds(ts: pd.Series) -> float | None:
    if len(ts) < 2:
        return None
    d = ts.sort_values().diff().dropna()
    if len(d) == 0:
        return None
    return float(d.dt.total_seconds().median())


def _seconds_on_boundary_ratio(ts: pd.Series, freq_s: int) -> float:
    # For 60s grid: timestamp should land on :00 seconds.
    # For general freq_s, require (epoch_seconds % freq_s)==0.
    #s = ts.view("int64") // 1_000_000_000

    s = ts.astype("int64") // 1_000_000_000

    ok = (s % int(freq_s)) == 0
    return float(ok.mean()) if len(ok) else 1.0






def standardize_bars_to_grid(
    bars: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    ts_col: str = "timestamp",
    freq: str = "60s",
    expected_freq_s: int = 60,
    global_index: pd.DatetimeIndex | None = None,
    fill_volume_zero: bool = True,
    keep_ohlc_nan: bool = True,
    hard_fail_on_duplicates: bool = False,
) -> tuple[pd.DataFrame, list[GridAuditRow]]:


    """
    Force every symbol onto one canonical UTC grid (default 60s).
    Invariant after this:
      - ts_col is tz-aware UTC
      - timestamps lie on the freq boundary
      - one row per (symbol, timestamp)
      - missing bars become explicit gaps (NaNs) after reindex
    """
    if ts_col not in bars.columns:
        raise ValueError(f"[Grid] missing required ts col '{ts_col}'")
    if symbol_col not in bars.columns:
        raise ValueError(f"[Grid] missing required symbol col '{symbol_col}'")

    out_parts: list[pd.DataFrame] = []
    audits: list[GridAuditRow] = []

    # Enforce tz-aware UTC on input (in-place OK)
    ensure_utc_timestamp_col(bars, ts_col, who="[Grid]")

    # Work symbol-by-symbol
    for sym, df in bars.groupby(symbol_col, sort=False):
        df = df.sort_values(ts_col).copy()
        n_in = len(df)

        med_in = _median_delta_seconds(df[ts_col])

        # Drop duplicates on timestamp (per symbol)
        dup_mask = df.duplicated(subset=[ts_col], keep="first")
        dup_n = int(dup_mask.sum())
        if dup_n:
            if hard_fail_on_duplicates:
                raise GridAuditError(f"[Grid] duplicates detected for symbol={sym}: {dup_n} rows share same {ts_col}")
            df = df.loc[~dup_mask].copy()

        # Construct full UTC index from min..max on canonical grid
        '''min_ts = pd.Timestamp(df[ts_col].min())
        max_ts = pd.Timestamp(df[ts_col].max())

        # Ensure tz-aware UTC stamps in index
        if min_ts.tz is None:
            min_ts = min_ts.tz_localize("UTC")
        else:
            min_ts = min_ts.tz_convert("UTC")
        if max_ts.tz is None:
            max_ts = max_ts.tz_localize("UTC")
        else:
            max_ts = max_ts.tz_convert("UTC")

        full_idx = pd.date_range(min_ts, max_ts, freq=freq, tz="UTC")
        '''

        # Construct the canonical UTC index.
        # IMPORTANT: if a global_index is provided, we use it (prevents 24h min..max expansion).
        if global_index is not None:
            full_idx = pd.DatetimeIndex(global_index)
            # ensure tz-aware UTC
            if getattr(full_idx, "tz", None) is None:
                full_idx = full_idx.tz_localize("UTC")
            else:
                full_idx = full_idx.tz_convert("UTC")
            # Define min/max for auditing when global_index is used
            min_ts = pd.Timestamp(full_idx.min())
            max_ts = pd.Timestamp(full_idx.max())

        else:
            # fallback: per-symbol min..max (can be very large if your data spans many days)
            '''min_ts = pd.Timestamp(df[ts_col].min())
            max_ts = pd.Timestamp(df[ts_col].max())

            if min_ts.tz is None:
                min_ts = min_ts.tz_localize("UTC")
            else:
                min_ts = min_ts.tz_convert("UTC")
            if max_ts.tz is None:
                max_ts = max_ts.tz_localize("UTC")
            else:
                max_ts = max_ts.tz_convert("UTC")

            full_idx = pd.date_range(min_ts, max_ts, freq=freq, tz="UTC")
            '''

            # IMPORTANT: DO NOT expand min→max with pd.date_range().
            # That creates a 24h clock and explodes missingness (fake gaps).
            # Instead, build the clock from OBSERVED timestamps (floored to the grid).
            ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce").dropna()
            if len(ts) == 0:
                full_idx = pd.DatetimeIndex([], tz="UTC")
                min_ts = pd.Timestamp("1970-01-01", tz="UTC")
                max_ts = pd.Timestamp("1970-01-01", tz="UTC")
            else:
                ts = ts.dt.floor(freq)
                full_idx = pd.DatetimeIndex(ts.unique()).sort_values()
                # debug: prove we didn't min→max expand
                print(
                    f"[Grid] fallback clock (observed) symbol={sym} n={len(full_idx)} min={full_idx.min() if len(full_idx) else '∅'} max={full_idx.max() if len(full_idx) else '∅'}")

                # ensure tz-aware UTC
                if getattr(full_idx, "tz", None) is None:
                    full_idx = full_idx.tz_localize("UTC")
                else:
                    full_idx = full_idx.tz_convert("UTC")

                # define min/max for auditing consistency
                min_ts = pd.Timestamp(full_idx.min())
                max_ts = pd.Timestamp(full_idx.max())

        # Reindex onto the grid
        df = df.set_index(ts_col).reindex(full_idx)
        df.index.name = ts_col

        # Restore columns
        df[symbol_col] = sym

        # --- Task2: missing-bar normalization ---
        # Rows introduced by reindexing should not create "partial bars".
        # If any OHLC is missing, treat the whole OHLC as missing.
        ohlc = [c for c in ("open", "high", "low", "close") if c in df.columns]
        if ohlc:
            any_missing = df[ohlc].isna().any(axis=1)
            df.loc[any_missing, ohlc] = np.nan


        # Fill policy
        if "volume" in df.columns and fill_volume_zero:
            df["volume"] = df["volume"].fillna(0)

        if not keep_ohlc_nan:
            # Only fill OHLC if you explicitly decide to; default is keep NaNs.
            for c in ("open", "high", "low", "close"):
                if c in df.columns:
                    df[c] = df[c].ffill()

        df = df.reset_index()

        # Output audits
        med_out = _median_delta_seconds(df[ts_col])
        n_out = len(df)
        missing_ratio = float(df.isna().any(axis=1).mean()) if n_out else 0.0

        tz_ok = pd.api.types.is_datetime64tz_dtype(df[ts_col]) and str(df[ts_col].dtype).endswith(", UTC]")
        boundary_ratio = _seconds_on_boundary_ratio(df[ts_col], expected_freq_s)

        audits.append(
            GridAuditRow(
                symbol=str(sym),
                expected_freq_s=int(expected_freq_s),
                min_ts=str(min_ts),
                max_ts=str(max_ts),
                n_rows_in=int(n_in),
                n_rows_out=int(n_out),
                duplicates_dropped=int(dup_n),
                median_delta_s_in=med_in,
                median_delta_s_out=med_out,
                missing_ratio_out=float(missing_ratio),
                tz_dtype_ok=bool(tz_ok),
                seconds_on_boundary_ratio=float(boundary_ratio),
            )
        )

        out_parts.append(df)

    out = pd.concat(out_parts, ignore_index=True)

    # Hard invariants (global)
    ensure_utc_timestamp_col(out, ts_col, who="[Grid OUT]")
    # duplicates per (symbol,timestamp) forbidden after standardization
    if out.duplicated(subset=[symbol_col, ts_col]).any():
        bad = int(out.duplicated(subset=[symbol_col, ts_col]).sum())
        raise GridAuditError(f"[Grid OUT] duplicates remain after standardization: {bad}")

    return out, audits


def grid_audit_to_json(audits: Sequence[GridAuditRow]) -> dict:
    return {"grid_audit": [asdict(a) for a in audits]}
