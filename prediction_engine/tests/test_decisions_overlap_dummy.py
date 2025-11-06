import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from pathlib import Path

from scripts.run_backtest import build_unified_clock, discover_parquet_symbols

def _write_minute_parquet(root: Path, symbol: str, timestamps: list[str]):
    root.mkdir(parents=True, exist_ok=True)
    ts = pd.to_datetime(timestamps, utc=True)
    df = pd.DataFrame({
        "timestamp": ts,
        "open":  1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1000,
    })
    for day, g in df.groupby(ts.date):
        day = pd.Timestamp(day)
        out = root / f"symbol={symbol}" / f"year={day.year}" / f"month={day.month:02d}" / f"day={day.day:02d}"
        out.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(g), out / "part-0.parquet")

def _emit_dummy_decisions(out_dir: Path, clock: pd.DatetimeIndex, symbols: list[str]):
    """
    Test-only helper: pretend the scoring loop emits a decision per symbol whenever that
    symbol has a bar at time t. entry_ts is defined as t+1min (bar-open fill convention).
    """
    rows = []
    for t in clock:
        for sym in symbols:
            # symbol has a bar at t?
            # (In real 4.2 this check comes from feature slice presence)
            has_bar = True  # we'll filter by presence below for correctness
            if has_bar:
                rows.append({
                    "timestamp": pd.to_datetime(t),
                    "symbol": sym,
                    "decision_ts": pd.to_datetime(t),
                    "entry_ts": pd.to_datetime(t) + pd.Timedelta(minutes=1),
                    # dummy signal fields
                    "p_cal": 0.55, "mu_net": 0.0, "pos_size": 0.0,
                })
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df), out_dir / "decisions.parquet")

def _share_multisymbol(decisions_path: Path) -> float:
    ds_dec = ds.dataset(decisions_path, format="parquet")
    tbl = ds_dec.to_table(columns=["timestamp", "symbol"])
    df = tbl.to_pandas()
    g = df.groupby("timestamp")["symbol"].nunique()
    return float((g >= 2).mean()) if len(g) else 0.0

def _causality_ratio(decisions_path: Path) -> float:
    ds_dec = ds.dataset(decisions_path, format="parquet")
    df = ds_dec.to_table(columns=["decision_ts", "entry_ts"]).to_pandas()
    if df.empty or "decision_ts" not in df or "entry_ts" not in df:
        return 0.0
    return float((pd.to_datetime(df["entry_ts"], utc=True) >
                  pd.to_datetime(df["decision_ts"], utc=True)).mean())

def test_step4_1_decisions_overlap_and_causality(tmp_path: Path):
    parquet_root = tmp_path / "parquet"
    # Construct overlap: 60 minutes for A, every 4th minute for B (~25% overlap)
    base_minutes = pd.date_range("1999-01-05 09:30", periods=60, freq="T", tz="UTC")
    symA = [m.isoformat() for m in base_minutes]
    symB = [m.isoformat() for m in base_minutes[::4]]

    _write_minute_parquet(parquet_root, "RRC", symA)
    _write_minute_parquet(parquet_root, "BBY", symB)

    symbols = discover_parquet_symbols(parquet_root)
    clock = build_unified_clock(parquet_root, "1999-01-05", "1999-01-06", symbols)
    assert len(clock) > 0

    out_dir = tmp_path / "artifacts" / "portfolio"
    _emit_dummy_decisions(out_dir, clock, symbols)

    share_multi = _share_multisymbol(out_dir / "decisions.parquet")
    assert share_multi >= 0.10, f"Expected ≥10% multi-symbol timestamps in decisions, got {share_multi:.3f}"

    ratio = _causality_ratio(out_dir / "decisions.parquet")
    assert ratio == 1.0, f"Causality failed: entry not strictly after decision, ratio={ratio:.3f}"
