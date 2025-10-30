from __future__ import annotations
import json, time
from pathlib import Path
import pandas as pd
from feature_engineering.pipelines.dataset_loader import load_slice
from feature_engineering.pipelines.core import CoreFeaturePipeline

def run_phase3_checks(input_root: str | Path, output_root: str | Path,
                      symbols: list[str], start: str, end: str,
                      normalization: str = "per_symbol") -> dict:
    t0 = time.perf_counter()
    df_raw, clock = load_slice(input_root, symbols, start, end)
    t1 = time.perf_counter()
    pipe = CoreFeaturePipeline(parquet_root=input_root)
    df, meta = pipe.run_mem(df_raw, normalization_mode=normalization)
    t2 = time.perf_counter()

    # Quantitative: output rowcount should match input rows (± any documented warmup rows—here we assume none)
    rows_in, rows_out = len(df_raw), len(df)
    row_ok = rows_in == rows_out

    # Throughput
    load_rate  = len(df_raw) / max(1e-9, (t1 - t0))
    fe_rate    = len(df) / max(1e-9, (t2 - t1))

    # Minimal write check (single-file option; if you use partitioned writer, adapt accordingly)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    out_file = output_root / "features.parquet"
    df.to_parquet(out_file, index=False)

    return {
        "input": {"symbols": symbols, "start": start, "end": end, "normalization": normalization},
        "counts": {"rows_in": rows_in, "rows_out": rows_out, "row_count_preserved": row_ok},
        "throughput": {"load_rows_per_sec": load_rate, "fe_rows_per_sec": fe_rate},
        "output_file": str(out_file.resolve()),
        "meta_mode": meta.get("normalization_mode")
    }

def pretty_print(report: dict) -> None:
    print(json.dumps(report, indent=2, default=str))
