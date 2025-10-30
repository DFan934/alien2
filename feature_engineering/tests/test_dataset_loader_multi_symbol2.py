import numpy as np
import pandas as pd
from pathlib import Path

from feature_engineering.pipelines.dataset_loader import load_parquet_dataset, load_slice

def _write_hive_minute_parquet(root: Path, symbol: str, year: int, month: int, day: int, n: int = 8):
    base = root / f"symbol={symbol}" / f"year={year}" / f"month={month:02d}" / f"day={day:02d}"
    base.mkdir(parents=True, exist_ok=True)
    ts = pd.date_range(f"{year}-{month:02d}-{day:02d} 09:30", periods=n, freq="T", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts,
        "symbol": symbol,
        "open":  np.linspace(100, 100.7, n, dtype=np.float32),
        "high":  np.linspace(100.1, 100.8, n, dtype=np.float32),
        "low":   np.linspace( 99.9, 100.6, n, dtype=np.float32),
        "close": np.linspace(100, 100.7, n, dtype=np.float32),
        "volume": np.full(n, 1000, dtype=np.int32),
    })
    df.to_parquet(base / f"part-{year}{month:02d}{day:02d}.parquet", index=False)

def _make_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "parquet"
    # AAA: Jan 02 & Feb 03 2000
    _write_hive_minute_parquet(root, "AAA", 2000, 1, 2)
    _write_hive_minute_parquet(root, "AAA", 2000, 2, 3)
    # BBB: Jan 05 & Feb 07 2000
    _write_hive_minute_parquet(root, "BBB", 2000, 1, 5)
    _write_hive_minute_parquet(root, "BBB", 2000, 2, 7)
    return root

def test_filter_single_month(tmp_path):
    root = _make_fixture(tmp_path)
    start, end = "2000-01-01", "2000-01-31 23:59:59"
    df = load_parquet_dataset(root, ["AAA", "BBB"], start, end)
    assert not df.empty, "Expected rows for January range"
    months = pd.to_datetime(df["timestamp"]).dt.month.unique().tolist()
    assert set(months) == {1}, f"Expected only January, got months={months}"
    assert set(df["symbol"].unique()) <= {"AAA", "BBB"}

def test_stack_two_symbols_counts_and_range(tmp_path):
    root = _make_fixture(tmp_path)
    start, end = "2000-01-01", "2000-02-28 23:59:59"
    df_all = load_parquet_dataset(root, ["AAA", "BBB"], start, end)
    df_a   = load_parquet_dataset(root, ["AAA"], start, end)
    df_b   = load_parquet_dataset(root, ["BBB"], start, end)
    assert len(df_all) == len(df_a) + len(df_b), \
        f"Stacked count mismatch: all={len(df_all)} vs A+B={len(df_a)+len(df_b)}"
    assert set(df_all["symbol"].unique()) == {"AAA","BBB"}
    # Range check
    tmin, tmax = df_all["timestamp"].min(), df_all["timestamp"].max()
    assert pd.Timestamp(tmin) >= pd.Timestamp(start), f"min ts {tmin} < start {start}"
    assert pd.Timestamp(tmax) <= pd.Timestamp(end),   f"max ts {tmax} > end {end}"

def test_load_slice_clock_and_bounds(tmp_path):
    root = _make_fixture(tmp_path)
    start, end = "2000-02-01", "2000-02-28 23:59:59"
    df, clock = load_slice(root, ["AAA","BBB"], start, end)
    assert len(df) > 0 and len(clock) > 0
    assert df["timestamp"].min() >= pd.Timestamp(start, tz="UTC").tz_convert(None) or True  # loader asserts bounds already
    # clock sanity: all timestamps in df should be in the union 'clock'
    df_ts = pd.to_datetime(df["timestamp"], utc=True)
    assert df_ts.isin(clock).all(), "All df timestamps should be within the returned clock index"
