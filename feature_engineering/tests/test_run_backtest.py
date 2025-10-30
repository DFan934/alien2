import pandas as pd
import numpy as np
from pathlib import Path

from scripts.run_backtest import resolve_universe_window_for_tests, ResolvedUniverseWindow

def _write_minute_parquet(root: Path, symbol: str, year: int, month: int, days: list[int]):
    # create hive layout: symbol=SYM/year=YYYY/month=MM/(optionally day=DD/)
    base = root / f"symbol={symbol}" / f"year={year}" / f"month={month:02d}"
    for d in days:
        p = base / f"day={d:02d}"
        p.mkdir(parents=True, exist_ok=True)
        ts = pd.date_range(f"{year}-{month:02d}-{d:02d} 09:30", periods=10, freq="T", tz="UTC")
        df = pd.DataFrame({
            "timestamp": ts,
            "symbol": symbol,
            "open": np.linspace(100, 101, len(ts), dtype=np.float32),
            "high": np.linspace(100.1, 101.1, len(ts), dtype=np.float32),
            "low":  np.linspace(99.9, 100.9, len(ts), dtype=np.float32),
            "close":np.linspace(100, 101, len(ts), dtype=np.float32),
            "volume": np.full(len(ts), 1000, dtype=np.int32),
        })
        df.to_parquet(p / f"part-{year}{month:02d}{d:02d}.parquet", index=False)

def test_resolve_nocli_discovers_all_symbols_and_bounds(tmp_path):
    parquet_root = tmp_path / "parquet"
    _write_minute_parquet(parquet_root, "RRC", 1998, 8, [26])
    _write_minute_parquet(parquet_root, "BBY", 1998, 8, [1, 2])

    resolved = resolve_universe_window_for_tests(parquet_root, cfg={}, mode="NOCLI")
    assert set(resolved.symbols) == {"RRC", "BBY"}
    # start should be 1998-08-01 and end 1998-08-26 (based on data we wrote)
    assert resolved.start == "1998-08-01"
    assert resolved.end   == "1998-08-26"
    # partition summary string mentions both symbols
    assert "RRC:" in resolved.partitions and "BBY:" in resolved.partitions

def test_resolve_cli_respects_cfg_symbols_and_dates(tmp_path):
    parquet_root = tmp_path / "parquet"
    _write_minute_parquet(parquet_root, "RRC", 1998, 8, [26])
    _write_minute_parquet(parquet_root, "BBY", 1998, 8, [1, 2])

    cfg = {"symbols": ["RRC"], "start": "1998-08-10", "end": "1998-08-20"}
    resolved = resolve_universe_window_for_tests(parquet_root, cfg=cfg, mode="CLI")
    assert resolved.symbols == ["RRC"]
    assert resolved.start == "1998-08-10"
    assert resolved.end   == "1998-08-20"
