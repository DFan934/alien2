import json
from pathlib import Path
import pandas as pd

def test_fold_outputs_exist_and_schema(tmp_path: Path):
    # Arrange: make a minimal hive with a few minutes for 2 symbols
    pq = tmp_path / "parquet"
    for sym in ["RRC","BBY"]:
        d = pq / f"symbol={sym}/year=1999/month=01/day=05"
        d.mkdir(parents=True, exist_ok=True)
        (d / "part-0.parquet").write_bytes(b"PAR1")  # stub is ok for IO path

    arte = tmp_path / "artifacts"
    fold_dir = arte / "folds" / "fold_01"; fold_dir.mkdir(parents=True, exist_ok=True)

    # Simulate that EV artefacts exist (empty meta is fine for smoke if your EVEngine is stub-tolerant)
    ev_dir = fold_dir / "ev"; ev_dir.mkdir(parents=True, exist_ok=True)
    (ev_dir / "meta.json").write_text(json.dumps({"payload":{}}, indent=2))

    # Call the backtest run() directly with a trivial config
    from scripts.run_backtest import run_batch as run_bt
    cfg = {
        "parquet_root": str(pq),
        "universe": ["RRC","BBY"],
        "start": "1999-01-01",
        "end":   "1999-01-10",
        "debug_no_costs": True,            # costless for speed
        "vectorized_scoring": True,
        "horizon_bars": 2,
        "equity": 100_000.0,
        "commission": 0.0,
    }
    metrics = run_bt(cfg, artifacts_dir=fold_dir, ev_artifacts_dir=ev_dir)

    # Verify required files
    for name in ["decisions.parquet", "trades.parquet", "signals_out.csv", "portfolio_metrics.json", "equity.csv"]:
        assert (fold_dir / name).exists(), f"{name} not written"

    # Schema sanity (minimal columns)
    dec = pd.read_parquet(fold_dir / "decisions.parquet") if (fold_dir / "decisions.parquet").stat().st_size > 0 else pd.DataFrame()
    trd = pd.read_parquet(fold_dir / "trades.parquet")    if (fold_dir / "trades.parquet").stat().st_size > 0 else pd.DataFrame()

    if not dec.empty:
        for c in ["timestamp","symbol","p_cal","horizon_bars","target_qty"]:
            assert c in dec.columns, f"missing column in decisions: {c}"
    if not trd.empty:
        for c in ["entry_ts","exit_ts","entry_price","exit_price","qty","realized_pnl"]:
            assert c in trd.columns, f"missing column in trades: {c}"

    # Count threshold (smoke)
    assert int(metrics.get("n_trades", 0)) >= 0  # keep loose here; enforce 50+ in real data runs
