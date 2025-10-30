import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pathlib import Path

from feature_engineering.pipelines.dataset_loader import (
    open_parquet_dataset,
    load_parquet_dataset,
    load_slice,
)

def _write_hive(tmp: Path, symbol: str, year: int, month: int, rows: int = 5):
    # Simple 1-minute bars
    ts = pd.date_range(f"{year}-{month:02d}-01 09:30", periods=rows, freq="T", tz="UTC")
    tbl = pa.table({
        #"timestamp": pa.array(ts.astype("datetime64[ns]")),
        "timestamp": pa.array(ts, type=pa.timestamp("ns", tz="UTC")),

        "open": pa.array([1.0]*rows, type=pa.float32()),
        "high": pa.array([1.1]*rows, type=pa.float32()),
        "low":  pa.array([0.9]*rows, type=pa.float32()),
        "close":pa.array([1.0]*rows, type=pa.float32()),
        "volume": pa.array([100]*rows, type=pa.int32()),
        "symbol": pa.array([symbol]*rows),
    })
    outdir = tmp / f"symbol={symbol}/year={year}/month={month:02d}"
    outdir.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, outdir / "part-0.parquet")

def test_multi_symbol_stack_and_filter(tmp_path: Path):
    _write_hive(tmp_path, "RRC", 1998, 8, rows=7)
    _write_hive(tmp_path, "BBY", 1998, 8, rows=9)

    # Sanity: dataset opens and carries hive partition schema
    ds_obj = open_parquet_dataset(tmp_path)
    assert isinstance(ds_obj, ds.Dataset)

    # Load both symbols in a constrained window
    start, end = "1998-08-01 09:30:00Z", "1998-08-01 09:36:00Z"
    df = load_parquet_dataset(tmp_path, ["RRC","BBY"], start, end)

    # Correctness: length equals sum of per-symbol lengths if loaded individually
    df_rrc = load_parquet_dataset(tmp_path, ["RRC"], start, end)
    df_bby = load_parquet_dataset(tmp_path, ["BBY"], start, end)
    assert len(df) == len(df_rrc) + len(df_bby)

    # Schema/dtypes
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert df["timestamp"].dt.tz is not None  # tz-aware
    for c in ["open","high","low","close"]:
        assert df[c].dtype == "float32"
    assert df["volume"].dtype == "int32"
    assert df["symbol"].dtype == object

def test_load_slice_returns_union_clock(tmp_path: Path):
    _write_hive(tmp_path, "RRC", 1998, 8, rows=3)
    _write_hive(tmp_path, "BBY", 1998, 8, rows=5)

    df, clock = load_slice(tmp_path, ["RRC","BBY"], "1998-08-01 09:30Z", "1998-08-01 09:34Z")
    # Union clock should span exactly the unique timestamps present
    assert len(clock) == df["timestamp"].nunique()
    assert clock.tz is not None
    assert clock.name == "timestamp"
