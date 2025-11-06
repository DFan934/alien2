# prediction_engine/scoring/tests/test_runner_parity.py
import asyncio
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scripts.run_backtest import run as run_bt  # async

def _write_minute_parquet(root: Path, symbol: str, ts_utc: pd.DatetimeIndex):
    """
    Write a minimal hive for one symbol with tz-aware UTC timestamps.
    Layout: symbol=SYM/year=YYYY/month=MM/day=DD/part-0.parquet
    """
    df = pd.DataFrame({
        "timestamp": ts_utc,   # tz-aware (UTC)
        "open":  10.0,
        "high":  10.0,
        "low":   10.0,
        "close": 10.0,
        "volume": 1000,
    })
    # partition by day
    for day, g in df.groupby(df["timestamp"].dt.date):
        d = pd.Timestamp(day)
        out = root / f"symbol={symbol}" / f"year={d.year}" / f"month={d.month:02d}" / f"day={d.day:02d}"
        out.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(g), out / "part-0.parquet")

def _build_tiny_hive(dest: Path):
    """
    Create two overlapping symbols (RRC, BBY) for 1999-01-05 so the runner has data.
    """
    dest.mkdir(parents=True, exist_ok=True)
    base = pd.date_range("1999-01-05 09:30", periods=60, freq="T", tz="UTC")
    _write_minute_parquet(dest, "RRC", base)          # every minute
    _write_minute_parquet(dest, "BBY", base[::4])     # every 4th minute

def _run(cfg_overrides: dict, tmp_path: Path) -> pd.DataFrame:
    arte = tmp_path / "artifacts"
    hive = tmp_path / "tiny_hive"
    _build_tiny_hive(hive)                            # <- make sure parquet_root exists

    cfg = {
        "equity": 100_000.0,
        "vectorized_scoring": False,                  # will override below
        "debug_no_costs": True,                       # isolate scorer parity
        "parquet_root": str(hive),
        "universe": ["RRC", "BBY"],
        "start": "1999-01-05",
        "end":   "1999-01-06",
        "artifacts_root": str(arte),
    }
    cfg.update(cfg_overrides)

    # run async entrypoint
    asyncio.run(run_bt(cfg))

    dec_path = arte / "portfolio" / "decisions.parquet"
    assert dec_path.exists(), f"decisions file not found at {dec_path}"
    return pd.read_parquet(dec_path).sort_values(["timestamp", "symbol"]).reset_index(drop=True)

def test_runner_decisions_identical_between_modes(tmp_path: Path):
    dec_loop = _run({"vectorized_scoring": False}, tmp_path)
    dec_vec  = _run({"vectorized_scoring": True},  tmp_path)
    pd.testing.assert_frame_equal(dec_loop, dec_vec, check_exact=True)
