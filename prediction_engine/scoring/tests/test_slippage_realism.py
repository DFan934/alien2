import pandas as pd
from pathlib import Path
from scripts.run_backtest import _promotion_checks_step4_8

def test_4_8_slippage_bps_threshold(tmp_path):
    port = tmp_path / "artifacts" / "a2" / "portfolio"
    port.mkdir(parents=True, exist_ok=True)

    t0 = pd.Timestamp("1999-01-05T09:30Z")

    # decisions: minimal, to trip sizing sanity
    decisions = pd.DataFrame({
        "timestamp": [t0, t0],
        "symbol": ["RRC","BBY"],
        "p_cal": [0.65, 0.70],
        "target_qty": [1000.0, 500.0],
    })
    decisions.to_parquet(port / "decisions.parquet", index=False)

    # trades consistent with MOO/MOC banding — error should be ~0 bps
    trades = pd.DataFrame({
        "symbol": ["RRC","BBY"],
        "entry_ts": [t0 + pd.Timedelta(minutes=1)]*2,
        "entry_price": [10.01, 20.02],  # open + half_spread
        "exit_ts":  [t0 + pd.Timedelta(minutes=3)]*2,
        "exit_price":[10.09, 20.06],    # open -/+ half_spread style; still consistent for median
        "qty": [1000, 500],
        "realized_pnl": [80.0, 40.0],
        "half_spread_usd": [0.01, 0.02],  # used by the checker
        "modeled_cost_total": [4.0, 2.0], # commission present
    })
    trades.to_parquet(port / "trades.parquet", index=False)

    (port / "equity_curve.csv").write_text("exit_ts,equity\n1999-01-05T09:33Z,120\n")
    (port / "portfolio_metrics.json").write_text('{"n_trades":2,"win_rate":1.0,"turnover":0.02,"max_drawdown":0.0}')

    out = _promotion_checks_step4_8(port_dir=port, cfg={"commission": 0.005})
    assert out["slippage_median_bps"] < 10.0
