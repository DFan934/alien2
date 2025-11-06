import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from scripts.run_backtest import run as run_bt  # adapt if your entry differs

def _write_day(root: Path, sym: str, ts: list[pd.Timestamp], open_=float):
    # minimal bars: open + volume; minute grid already UTC
    df = pd.DataFrame({"timestamp": ts, "open": open_, "high": open_, "low": open_, "close": open_, "volume": 60_000})
    for day, g in df.groupby(df["timestamp"].dt.date):
        d = pd.Timestamp(day)
        out = root / f"symbol={sym}/year={d.year}/month={d.month:02d}/day={d.day:02d}"
        out.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(g), out / "part-0.parquet")

def test_runner_writes_portfolio_artifacts(tmp_path):
    parquet_root = tmp_path / "parquet"
    ts = pd.date_range("1999-01-05 09:30", periods=6, freq="T", tz="UTC")
    _write_day(parquet_root, "RRC", ts, 10.0)
    _write_day(parquet_root, "BBY", ts, 20.0)

    artifacts = tmp_path / "artifacts"
    cfg = {
        "parquet_root": str(parquet_root),
        "artifacts_root": str(artifacts),
        "universe": ["RRC","BBY"],
        "start": "1999-01-05",
        "end":   "1999-01-06",
        "vectorized_scoring": True,
        "debug_no_costs": True,   # isolate wiring; costs already tested in 4.2
        "p_gate_quantile": 0.50,  # force a few entries
        "full_p_quantile": 0.60,
        "equity": 100_000.0,
    }

    run_bt(cfg)

    port = artifacts / "a2" / "portfolio"
    assert (port / "decisions.parquet").exists()
    assert (port / "trades.parquet").exists()
    assert (port / "equity_curve.csv").exists()
    assert (port / "portfolio_metrics.json").exists()
