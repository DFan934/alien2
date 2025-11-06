import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from pathlib import Path

# Import from the actual runner module
from scripts.run_backtest import build_unified_clock, discover_parquet_symbols  # noqa: E402

def _write_minute_parquet(root: Path, symbol: str, timestamps: list[str]):
    root.mkdir(parents=True, exist_ok=True)
    ts = pd.to_datetime(timestamps, utc=True)
    df = pd.DataFrame({
        "timestamp": ts,
        "open":  1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1000,
    })
    # Write under hive partitions: symbol=SYM/year=YYYY/month=MM/day=DD
    for day, g in df.groupby(ts.date):
        day = pd.Timestamp(day)
        out = root / f"symbol={symbol}" / f"year={day.year}" / f"month={day.month:02d}" / f"day={day.day:02d}"
        out.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(g), out / "part-0.parquet")

def test_unified_clock_overlap(tmp_path: Path):
    parquet_root = tmp_path / "parquet"
    # Symbol A has 60 minutes; Symbol B overlaps on 15 of them
    base_minutes = pd.date_range("1999-01-05 09:30", periods=60, freq="T", tz="UTC")
    symA = [m.isoformat() for m in base_minutes]
    symB = [m.isoformat() for m in base_minutes[::4]]  # every 4th minute overlaps (≈25% of minutes)

    _write_minute_parquet(parquet_root, "RRC", symA)
    _write_minute_parquet(parquet_root, "BBY", symB)

    symbols = discover_parquet_symbols(parquet_root)
    assert set(symbols) == {"BBY", "RRC"}

    clock = build_unified_clock(parquet_root, "1999-01-05", "1999-01-06", symbols)
    assert len(clock) == len(set(symA) | set(symB))  # union of minutes

    # Compute share of timestamps with >= 2 symbols present (acceptance ≥ 0.10 for 2–3 symbol tests)
    # (This mirrors the sanity print in run_backtest.py)
    present = []
    for t in clock:
        c = 0
        for sym in symbols:
            sym_dir = parquet_root / f"symbol={sym}"
            ds_sym = ds.dataset(str(sym_dir), format="parquet", partitioning="hive", exclude_invalid_files=True)
            #tbl = ds_sym.to_table(columns=["timestamp"], filter=(ds.field("timestamp") == pd.to_datetime(t)))
            t_type = ds_sym.schema.field("timestamp").type  # timestamp(unit, tz)
            tz = t_type.tz
            t_py = pd.to_datetime(t, utc=(tz is not None)).to_pydatetime()
            t_scalar = pa.scalar(t_py, type=t_type)
            tbl = ds_sym.to_table(columns=["timestamp"], filter=(ds.field("timestamp") == t_scalar))

            c += int(tbl.num_rows > 0)
        present.append(c >= 2)
    share_multi = float(pd.Series(present).mean())

    assert share_multi >= 0.10, f"Expected ≥10% multi-symbol timestamps, got {share_multi:.3f}"
