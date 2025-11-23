# prediction_engine/tests/test_fold_metrics_persist.py
import json
from pathlib import Path
import pandas as pd
import numpy as np

from prediction_engine.testing_validation.fold_metrics import compute_and_save_fold_metrics

def test_fold_metrics_files_and_sanity(tmp_path: Path):
    fold = tmp_path / "fold_00"; (fold / "metrics/plots").mkdir(parents=True, exist_ok=True)

    # --- minimal decisions/trades with simple structure
    # 100 synthetic decisions with p in [0,1], half wins, sprinkled regimes/sides
    n = 100
    rng = np.random.default_rng(42)
    decisions = pd.DataFrame({
        "timestamp": pd.date_range("2000-01-01", periods=n, freq="T"),
        "symbol": ["RRC"] * n,
        "p_cal": rng.uniform(0, 1, size=n),
        "regime": rng.choice(["TREND","RANGE","VOL","GLOBAL"], size=n, replace=True),
        "side": rng.choice(["long","short"], size=n, replace=True),
        "decision_id": np.arange(n),
        "nn_mu": rng.uniform(0, 1, size=n),
        "fallback": rng.choice([True, False], size=n, replace=True),
    })
    trades = pd.DataFrame({
        "decision_id": np.arange(n),
        "symbol": ["RRC"] * n,
        "entry_ts": decisions["timestamp"],
        "realized_pnl_after_costs": rng.choice([+1.0, -1.0], size=n)
    })

    dec_pq = fold / "decisions.parquet"
    trd_pq = fold / "trades.parquet"
    decisions.to_parquet(dec_pq, index=False)
    trades.to_parquet(trd_pq, index=False)

    payload = compute_and_save_fold_metrics(
        decisions_parquet=dec_pq,
        trades_parquet=trd_pq,
        out_dir=fold / "metrics",
    )

    # files exist
    fm = fold / "metrics" / "fold_metrics.json"
    assert fm.exists(), "fold_metrics.json not written"
    for p in ["reliability.png","decile_lift.png","residual_cusum.png"]:
        assert (fold / "metrics" / "plots" / p).exists(), f"{p} not written"

    data = json.loads(fm.read_text())
    # sanity ranges
    ece = data["ece"]; brier = data["brier"]; deciles = data["deciles"]
    assert 0 <= ece <= 1, f"ECE out of range: {ece}"
    assert 0 <= brier <= 1, f"Brier out of range: {brier}"
    assert 1 <= deciles <= 10, f"Decile count invalid: {deciles}"
