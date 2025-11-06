import pandas as pd
from pathlib import Path
from scripts.run_backtest import _promotion_checks_step4_8

def test_4_8_presence_and_multisymbol(tmp_path):
    port = tmp_path / "artifacts" / "a2" / "portfolio"
    port.mkdir(parents=True, exist_ok=True)

    # decisions: same minute, two symbols => multisymbol share >= 0.5
    t0 = pd.Timestamp("1999-01-05T09:30Z")
    decisions = pd.DataFrame({
        "timestamp": [t0, t0, t0 + pd.Timedelta(minutes=1)],
        "symbol": ["RRC","BBY","RRC"],
        "p_cal": [0.62, 0.71, 0.55],
        "target_qty": [1000.0, 500.0, 0.0],
    })
    decisions.to_parquet(port / "decisions.parquet", index=False)

    # trades: include spread + modeled_cost_total>0 (commission present)
    trades = pd.DataFrame({
        "symbol": ["RRC","BBY"],
        "entry_ts": [t0 + pd.Timedelta(minutes=1)]*2,
        "entry_price": [10.00, 20.00],
        "exit_ts":  [t0 + pd.Timedelta(minutes=3)]*2,
        "exit_price":[10.05, 20.05],
        "qty": [1000, 500],
        "realized_pnl": [50.0, 25.0],
        "half_spread_usd": [0.01, 0.02],
        "modeled_cost_total": [5.0, 2.5],
    })
    trades.to_parquet(port / "trades.parquet", index=False)

    # equity + metrics stubs just to satisfy presence gate
    (port / "equity_curve.csv").write_text("exit_ts,equity\n1999-01-05T09:33Z,75\n")
    (port / "portfolio_metrics.json").write_text('{"n_trades":2,"win_rate":1.0,"turnover":0.03,"max_drawdown":0.0}')

    cfg = {"commission": 0.005, "p_gate_quantile": 0.55}
    out = _promotion_checks_step4_8(port_dir=port, cfg=cfg)

    assert out["exists_decisions"] and out["exists_trades"] and out["exists_equity"] and out["exists_metrics_json"]
    assert out["share_multisymbol"] >= 0.10
    assert out["commission_nonzero"] is True
    assert out["sizing_median_pos_gt0"] is True
    assert out["sizing_cost_hurdle_violations"] == 0
