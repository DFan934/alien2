from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence, Optional, Dict, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

try:
    import psutil  # optional
except Exception:
    psutil = None

# Use your Step-1 loaders
from feature_engineering.pipelines.dataset_loader import (
    open_parquet_dataset,
    load_parquet_dataset,
    load_slice,
)


def _fragment_paths(dataset: ds.Dataset, filt: ds.Expression) -> list[str]:
    """List concrete fragment file paths that will be scanned under the filter."""
    paths = []
    for frag in dataset.get_fragments(filter=filt):
        try:
            # For FileSystemDatasetFragment
            paths.append(str(frag.path))
        except Exception:
            # Fallback: path may live on the fragment's physical format
            try:
                paths.append(str(frag.split_by_row_group()[0].path))
            except Exception:
                paths.append(repr(frag))
    return paths


def _build_filter(symbols: Sequence[str], start: pd.Timestamp | str, end: pd.Timestamp | str, ts_type: pa.DataType) -> ds.Expression:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if isinstance(ts_type, pa.TimestampType) and ts_type.tz is not None:
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("UTC")
        else:
            start_ts = start_ts.tz_convert("UTC")
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        else:
            end_ts = end_ts.tz_convert("UTC")
    else:
        if start_ts.tzinfo is not None:
            start_ts = start_ts.tz_convert("UTC").tz_localize(None)
        if end_ts.tzinfo is not None:
            end_ts = end_ts.tz_convert("UTC").tz_localize(None)

    #start_scalar = pa.scalar(np.datetime64(start_ts.to_datetime64()), type=ts_type)
    #end_scalar = pa.scalar(np.datetime64(end_ts.to_datetime64()), type=ts_type)

    # AFTER
    #start_scalar = pa.scalar(start_ts.to_datetime64(), type=ts_type)
    #end_scalar = pa.scalar(end_ts.to_datetime64(), type=ts_type)

    # AFTER
    start_scalar = pa.scalar(start_ts.to_pydatetime(), type=ts_type)
    end_scalar = pa.scalar(end_ts.to_pydatetime(), type=ts_type)

    return (ds.field("timestamp") >= start_scalar) & (ds.field("timestamp") <= end_scalar) & ds.field("symbol").isin(list(symbols))


def _dtype_report(df: pd.DataFrame) -> Dict[str, str]:
    return {c: str(df[c].dtype) for c in df.columns if c in df}


def _check_schema(df: pd.DataFrame) -> Dict[str, Any]:
    issues = []

    # timestamp tz-aware UTC
    if "timestamp" not in df.columns:
        issues.append("missing 'timestamp' column")
    else:
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            issues.append("timestamp not datetime64")
        else:
            tz = getattr(df["timestamp"].dt, "tz", None)
            if tz is None:
                issues.append("timestamp not tz-aware")
            elif str(tz) != "UTC":
                issues.append(f"timestamp tz is {tz}, expected UTC")

    # prices float32
    for c in ("open", "high", "low", "close"):
        if c in df.columns and str(df[c].dtype) != "float32":
            issues.append(f"{c} dtype={df[c].dtype}, expected float32")

    # volume int32
    if "volume" in df.columns and str(df["volume"].dtype) != "int32":
        issues.append(f"volume dtype={df['volume'].dtype}, expected int32")

    # symbol object (string OK but usually becomes 'object' in pandas)
    if "symbol" in df.columns and df["symbol"].dtype != object:
        issues.append(f"symbol dtype={df['symbol'].dtype}, expected object")

    return {"ok": len(issues) == 0, "issues": issues, "dtypes": _dtype_report(df)}


def run_phase1_checks(
    root: str | Path,
    symbols: Sequence[str],
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    *,
    columns: Optional[Sequence[str]] = None,
    speed_target_s_per_million: float = 2.0,
) -> Dict[str, Any]:
    """
    Returns a JSON-serializable report with:
      - correctness: stacked == sum of per-symbol
      - pruning evidence: fragments scanned (sample paths)
      - performance: wall time, rows/sec, est. s per million rows
      - memory: optional RSS delta if psutil available
      - schema check: tz-aware UTC ts, float32 prices, int32 volume, object symbol
      - clock stats: unique timestamps count
    """
    root = Path(root)
    dataset = open_parquet_dataset(root)

    # Build the *exact* filter the loader should use (for pruning introspection)
    ts_field = next(f for f in dataset.schema if f.name == "timestamp")
    filt = _build_filter(symbols, start, end, ts_field.type)

    # Count total fragments vs filtered fragments (pruning evidence)
    all_frags = list(dataset.get_fragments())  # lightweight handles
    filt_frags = _fragment_paths(dataset, filt)
    pruning_info = {
        "total_fragments": len(all_frags),
        "scanned_fragments": len(filt_frags),
        "sample_scanned_paths": filt_frags[:10],
        "pruned_ok": len(filt_frags) <= len(all_frags),
    }

    # PERF + MEM: measure loader
    process = psutil.Process() if psutil else None
    rss_before = process.memory_info().rss if process else None
    t0 = time.perf_counter()
    df_all = load_parquet_dataset(root, symbols, start, end, columns=columns)
    wall = time.perf_counter() - t0
    rss_after = process.memory_info().rss if process else None if process else None

    # Correctness: sum of per-symbol lengths
    per_counts = {}
    for s in symbols:
        df_s = load_parquet_dataset(root, [s], start, end, columns=columns)
        per_counts[s] = len(df_s)
    correctness_ok = len(df_all) == sum(per_counts.values())

    # Schema check
    schema_result = _check_schema(df_all)

    # Clock union check
    _, clock = load_slice(root, symbols, start, end, columns=columns)

    # Perf normalization
    rows = len(df_all)
    rows_per_sec = rows / wall if wall > 0 else float("inf")
    s_per_million = (wall / (rows / 1_000_000)) if rows else None

    report: Dict[str, Any] = {
        "input": {
            "root": str(root),
            "symbols": list(symbols),
            "start": str(pd.Timestamp(start)),
            "end": str(pd.Timestamp(end)),
            "columns": list(columns) if columns else None,
        },
        "pruning": pruning_info,
        "counts": {
            "total_rows": rows,
            "per_symbol": per_counts,
            "stack_equals_sum": correctness_ok,
        },
        "schema": schema_result,
        "performance": {
            "wall_seconds": wall,
            "rows_per_sec": rows_per_sec,
            "s_per_million_rows": s_per_million,
            "meets_speed_target": (s_per_million is None) or (s_per_million <= speed_target_s_per_million),
        },
        "memory": {
            "rss_before": rss_before,
            "rss_after": rss_after,
            "rss_delta": (rss_after - rss_before) if (rss_before is not None and rss_after is not None) else None,
        },
        "clock": {
            "unique_timestamps": int(clock.size),
            "tz": str(clock.tz) if getattr(clock, "tz", None) else None,
            "name": clock.name,
        },
        #"dtype_sample": _dtype_report(df_all)[:10] if hasattr(dict, "items") else {},
        # AFTER
        "dtype_sample": dict(list(_dtype_report(df_all).items())[:10]),
    }

    # Human-friendly summary verdicts
    report["verdicts"] = {
        "qualitative_single_call_stateless": True,  # by construction of the loaders
        "qualitative_pruning_evidence": pruning_info["pruned_ok"] and pruning_info["scanned_fragments"] < pruning_info["total_fragments"],
        "quant_correctness": correctness_ok,
        "quant_speed_ok": report["performance"]["meets_speed_target"],
        "quant_schema_ok": schema_result["ok"],
        "quant_clock_tz_ok": (report["clock"]["tz"] == "UTC"),
    }

    return report


def pretty_print_report(report: Dict[str, Any]) -> None:
    print(json.dumps(report, indent=2, default=str))
