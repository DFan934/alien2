# File: tests/test_feature_ordering.py

import pandas as pd
import pytest
from io import BytesIO

from scanner.backtest_loop import BacktestScannerLoop
from scanner.schema import FEATURE_ORDER
from feature_engineering.run_pipeline import main as run_pipeline_main
from tempfile import TemporaryDirectory
import os

def test_backtest_loop_emits_in_feature_order():
    # build a tiny DataFrame with out‐of‐order columns + dummy detector
    df = pd.DataFrame([
        {"symbol": "X", "timestamp": pd.Timestamp("2025-07-15T10:00Z"),
        ** {col: 0 for col in FEATURE_ORDER if col not in ("symbol", "timestamp")}}
    ])
    # ensure the DataFrame index is aligned to the timestamp values
    df.index = pd.DatetimeIndex(df["timestamp"])
    # dummy detector that matches everything
    class DummyDet:
        def __call__(self, slice_df):
            return pd.Series([True]*len(slice_df), index=slice_df.index)
    from scanner.recorder import DataGroupBuilder
    builder = DataGroupBuilder(parquet_root=".", buffer_size=1)
    loop = BacktestScannerLoop(DummyDet(), builder, df)
    # consume one event
    ts, sym, row = next(iter(loop))
    # row should have exactly FEATURE_ORDER as its index
    assert list(row.index) == list(FEATURE_ORDER)

def test_run_pipeline_writes_in_feature_order(tmp_path, monkeypatch):
    # create a tiny parquet input under the expected layout:
    input_root = tmp_path/"parquet"
    symbol = "ABC"
    part = input_root/f"symbol={symbol}"/"year=2025"/"month=07"
    part.mkdir(parents=True)

    # build a dummy raw‐bar file (open/high/low/close/volume + symbol/timestamp)
    raw = {
        "open": 1.0,
        "high": 1.1,
        "low": 0.9,
        "close": 1.05,
        "volume": 100.0,
        "symbol": symbol,
        # naive timestamp so Parquet date filtering works
        "timestamp": pd.Timestamp("2025-07-15T10:00"),
        # ✨ FIX: Add the missing 'trigger_ts' column
        "trigger_ts": pd.Timestamp("2025-07-15T10:00"),
        "volume_spike_pct": 0.0,
        }
    bar = pd.DataFrame([raw])
    bar.to_parquet(part/"test.parquet", index=False)
    # now run pipeline, pointing at input_root and an output dir
    monkeypatch.chdir(tmp_path)  # ensure relative paths resolve here
    import sys
    # simulate command-line args
    sys.argv = ["feature_engineering.run_pipeline", "--input_root", str(input_root), "--output_root", str(tmp_path/"out")]
    run_pipeline_main()
    # read back the output parquet
    out_dir = tmp_path / "out" / f"symbol={symbol}" / "year=2025" / "month=07"
    # there may be one or more parquet files; read them all and concat
    parts = list(out_dir.glob("*.parquet"))
    out = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    # strip off Hive partition columns (year, month) before comparing
    # Extract only the canonical FEATURE_ORDER columns, in the order they appear
    actual_features = [c for c in out.columns if c in FEATURE_ORDER]
    # Assert that their relative ordering matches exactly our canonical tuple
    assert actual_features == list(FEATURE_ORDER)