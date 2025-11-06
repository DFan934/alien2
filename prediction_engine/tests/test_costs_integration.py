import pandas as pd
from scripts.run_backtest import _simulate_trades_from_decisions, _apply_modeled_costs_to_trades  # noqa: E402

def test_simulated_trades_have_modeled_costs():
    ts = pd.date_range("1999-01-05 09:30", periods=5, freq="T", tz="UTC")
    bars = pd.DataFrame({
        "timestamp": list(ts) * 2,
        "symbol": ["RRC"]*5 + ["BBY"]*5,
        "open":   [10,10.1,10.2,10.3,10.4] + [20,20.1,20.2,20.3,20.4],
        "volume": [60_000]*10,
    })
    decisions = pd.DataFrame({
        "timestamp": [ts[1], ts[2]],
        "symbol": ["RRC", "BBY"],
        "target_qty": [10_000, 50_000],
        "horizon_bars": [2, 2],
    })
    trades = _simulate_trades_from_decisions(decisions, bars, rules={"max_participation": 0.25})
    assert not trades.empty
    out = _apply_modeled_costs_to_trades(trades, cfg={"debug_no_costs": False})
    assert (out["modeled_cost_total"] > 0).all()
    assert "realized_pnl_after_costs" in out.columns
