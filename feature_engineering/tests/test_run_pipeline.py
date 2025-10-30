import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from feature_engineering.pipelines.dataset_loader import load_slice
from feature_engineering.pipelines.core import CoreFeaturePipeline

# ---------------------------
# Helpers: fixture data + writer
# ---------------------------

def _write_hive_minute_parquet(root: Path, symbol: str, year: int, month: int, day: int, n: int = 12):
    """Create a small minute-bar parquet under hive layout:
       symbol=SYM/year=YYYY/month=MM/day=DD/part-YYYYMMDD.parquet
    """
    base = root / f"symbol={symbol}" / f"year={year}" / f"month={month:02d}" / f"day={day:02d}"
    base.mkdir(parents=True, exist_ok=True)
    ts = pd.date_range(f"{year}-{month:02d}-{day:02d} 09:30", periods=n, freq="T", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts,
        "symbol": symbol,
        "open":  np.linspace(100, 101, n, dtype=np.float32),
        "high":  np.linspace(100.1, 101.1, n, dtype=np.float32),
        "low":   np.linspace( 99.9, 100.9, n, dtype=np.float32),
        "close": np.linspace(100, 101, n, dtype=np.float32),
        "volume": np.full(n, 1_000, dtype=np.int32),
    })
    df.to_parquet(base / f"part-{year}{month:02d}{day:02d}.parquet", index=False)


def _build_tiny_universe(tmp_path: Path) -> Path:
    """Two months of data for three symbols over distinct days so filters are meaningful."""
    root = tmp_path / "parquet"
    # SYM1: Jan 02 & Feb 03 2000
    _write_hive_minute_parquet(root, "SYM1", 2000, 1, 2, n=16)
    _write_hive_minute_parquet(root, "SYM1", 2000, 2, 3, n=16)
    # SYM2: Jan 05 & Feb 07 2000
    _write_hive_minute_parquet(root, "SYM2", 2000, 1, 5, n=16)
    _write_hive_minute_parquet(root, "SYM2", 2000, 2, 7, n=16)
    # SYM3: Jan 09 & Feb 11 2000
    _write_hive_minute_parquet(root, "SYM3", 2000, 1, 9, n=16)
    _write_hive_minute_parquet(root, "SYM3", 2000, 2, 11, n=16)
    return root


def _run_acceptance_once(parquet_root: Path, symbols: list[str],
                         start: str, end: str, out_path: Path,
                         normalization: str = "per_symbol",
                         z_cols_hint: list[str] | None = None,
                         warmup_tolerance: int = 0) -> dict:
    """Replicates the Step-5 acceptance demo in-process (no CLI)."""
    # Load raw with our Step-1 loader
    df_raw, clock = load_slice(parquet_root, symbols, start, end)

    # Run FE (Step-2/3 paths), per_symbol normalization
    pipe = CoreFeaturePipeline(parquet_root=parquet_root)
    df_feat, meta = pipe.run_mem(df_raw, normalization_mode=normalization)

    # Persist a single stacked file (Step-3 writer behavior)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_parquet(out_path, index=False)

    # Re-read and validate
    df_out = pd.read_parquet(out_path)

    # 1) symbol integrity
    syms_out = sorted(df_out["symbol"].unique().tolist())
    syms_ok = syms_out == sorted(symbols)

    # 2) row-count ~= input rows (allowance for warmups if you later add rolling windows)
    rows_in, rows_out = len(df_raw), len(df_out)
    rows_ok = abs(rows_out - rows_in) <= warmup_tolerance

    # 3) per-symbol z-feature checks (if your pipeline writes standardized columns)
    if z_cols_hint:
        z_cols = [c for c in z_cols_hint if c in df_out.columns]
    else:
        z_cols = [c for c in df_out.columns if c.startswith("z_")]

    if z_cols:
        stats = df_out.groupby("symbol")[z_cols].agg(["mean", "std"])
        stats.columns = [f"{a}_{b}" for a, b in stats.columns]
        mu_max = float(stats.filter(like="_mean").abs().to_numpy().max())
        sd_max = float((stats.filter(like="_std") - 1.0).abs().to_numpy().max())
        z_ok = (mu_max < 0.05) and (sd_max < 0.05)
        z_info = {"max_abs_mean": mu_max, "max_abs_std_err": sd_max, "checked_cols": z_cols}
    else:
        # If only PCA features exist, skip z checks (covered by earlier unit tests).
        z_ok = True
        z_info = {"skipped": "no z_* columns present in output"}

    report = {
        "input": {"symbols": symbols, "start": start, "end": end, "normalization": normalization},
        "counts": {"rows_in": rows_in, "rows_out": rows_out, "row_count_preserved": rows_ok},
        "symbols": {"out": syms_out, "expected": symbols, "ok": syms_ok},
        "z_checks": z_info, "z_ok": z_ok,
        "meta_mode": meta.get("normalization_mode"),
        "out_file": str(out_path),
        "pass": syms_ok and rows_ok and z_ok and (meta.get("normalization_mode") == normalization),
    }
    return report


def _assert_report(report: dict):
    msg = json.dumps(report, indent=2)
    assert report["symbols"]["ok"], f"Symbol mismatch:\n{msg}"
    assert report["counts"]["row_count_preserved"], f"Rowcount mismatch beyond tolerance:\n{msg}"
    assert report["meta_mode"] == report["input"]["normalization"], f"Normalization mode mismatch:\n{msg}"
    assert report["z_ok"], f"Per-symbol z-checks failed (or supply z_cols_hint if needed):\n{msg}"
    assert report["pass"], f"Acceptance report not passing:\n{msg}"


# ---------------------------
# Acceptance tests: two universes
# ---------------------------

@pytest.mark.fast
def test_step5_acceptance_universe_A(tmp_path):
    root = _build_tiny_universe(tmp_path)
    out = tmp_path / "features" / "sym1_sym2_2000H1.parquet"
    report = _run_acceptance_once(
        parquet_root=root,
        symbols=["SYM1", "SYM2"],
        start="2000-01-01", end="2000-02-28 23:59:59",
        out_path=out,
        normalization="per_symbol",
        z_cols_hint=None,           # set to your z_* columns if you output them
        warmup_tolerance=0
    )
    _assert_report(report)


@pytest.mark.fast
def test_step5_acceptance_universe_B(tmp_path):
    root = _build_tiny_universe(tmp_path)
    out = tmp_path / "features" / "sym2_sym3_2000H1.parquet"
    report = _run_acceptance_once(
        parquet_root=root,
        symbols=["SYM2", "SYM3"],
        start="2000-01-01", end="2000-02-28 23:59:59",
        out_path=out,
        normalization="per_symbol",
        z_cols_hint=None,
        warmup_tolerance=0
    )
    _assert_report(report)
