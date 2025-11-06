import pandas as pd
from scripts.run_backtest import _apply_modeled_costs_to_trades  # noqa: E402

def _total_pnl_after_costs(df: pd.DataFrame) -> float:
    # stand-in for equity curve difference check
    return float(df["realized_pnl_after_costs"].sum()) if "realized_pnl_after_costs" in df else float(df["realized_pnl"].sum())

def test_equity_changes_when_costs_toggle():
    trades = pd.DataFrame({
        "symbol": ["RRC", "BBY", "RRC"],
        "entry_price": [10.0, 20.0, 10.5],
        "exit_price":  [10.1, 19.9, 10.6],
        "qty": [1000, 500, 800],
        "realized_pnl": [100.0, -50.0, 80.0],
        "half_spread_usd": [0.01, 0.02, None],
        "adv_frac": [0.05, 0.02, 0.03],
    })

    out_on  = _apply_modeled_costs_to_trades(trades, cfg={"debug_no_costs": False})
    out_off = _apply_modeled_costs_to_trades(trades, cfg={"debug_no_costs": True})

    # Costs ON must reduce total PnL
    assert _total_pnl_after_costs(out_on) < _total_pnl_after_costs(out_off)
