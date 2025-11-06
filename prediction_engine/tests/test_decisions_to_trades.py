import pandas as pd
from scripts.run_backtest import _simulate_trades_from_decisions  # noqa: E402

def test_decisions_to_trades_schema_and_behavior():
    # Minimal synthetic bars for two symbols
    ts = pd.date_range("1999-01-05 09:30", periods=6, freq="T", tz="UTC")
    bars = pd.DataFrame({
        "timestamp": list(ts) * 2,
        "symbol": ["RRC"] * 6 + ["BBY"] * 6,
        "open":   [10,10.1,10.2,10.3,10.4,10.5] + [20,20.1,20.2,20.3,20.4,20.5],
        "volume": [60_000]*12,
    })

    decisions = pd.DataFrame({
        "timestamp": [ts[1], ts[2]],  # decisions at t1,t2
        "symbol": ["RRC", "BBY"],
        "target_qty": [10_000, 50_000],   # one fully fillable, one capped
        "horizon_bars": [3, 2],
    })

    rules = {"max_participation": 0.25, "moo_gap_band": True, "moc_gap_band": True}
    trades = _simulate_trades_from_decisions(decisions, bars, rules=rules)

    required = {"symbol","entry_ts","entry_price","exit_ts","exit_price","bars_held","qty","realized_pnl","half_spread_usd","adv_frac"}
    assert required.issubset(trades.columns)

    # RRC: should fill ≈ 100% (cap 15k > target 10k)
    rrc = trades.loc[trades["symbol"] == "RRC"].iloc[0]
    assert rrc["qty"] > 9000
    # BBY: should be capped (cap 15k < target 50k)
    bby = trades.loc[trades["symbol"] == "BBY"].iloc[0]
    assert bby["qty"] <= 25_000
