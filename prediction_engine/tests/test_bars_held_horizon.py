import pandas as pd
from scripts.run_backtest import _simulate_trades_from_decisions  # noqa: E402

def test_bars_held_matches_horizon():
    ts = pd.date_range("1999-01-05 09:30", periods=8, freq="T", tz="UTC")
    bars = pd.DataFrame({
        "timestamp": list(ts) * 1,
        "symbol": ["RRC"] * len(ts),
        "open":   [10,10.1,10.2,10.3,10.4,10.5,10.6,10.7],
        "volume": [50_000]*len(ts),
    })
    decisions = pd.DataFrame({
        "timestamp": [ts[1]],
        "symbol": ["RRC"],
        "target_qty": [10_000],
        "horizon_bars": [3],
    })
    trades = _simulate_trades_from_decisions(decisions, bars, rules={"max_participation": 1.0})
    t = trades.iloc[0]
    assert int(t["bars_held"]) == 3
    # entry is at ts[2] (next open), exit at ts[5] (3 bars later)
    assert str(pd.Timestamp(t["exit_ts"])) == str(ts[5])
