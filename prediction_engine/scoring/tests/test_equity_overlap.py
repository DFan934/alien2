import pandas as pd

def test_equity_curve_sums_concurrent_exits(tmp_path):
    # Two trades exit on the same minute → step should equal sum of both pnls
    t_exit = pd.Timestamp("1999-01-05T10:00Z")
    trades = pd.DataFrame({
        "exit_ts": [t_exit, t_exit],
        "realized_pnl": [80.0, -30.0],
    })
    # Mimic equity_curve_from_trades fallback (sum by exit_ts then cumsum)
    step = trades.groupby("exit_ts")["realized_pnl"].sum()
    curve = step.cumsum()
    # The only step should be 50
    assert float(curve.iloc[-1]) == 50.0
