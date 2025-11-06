import pandas as pd
from scripts.run_backtest import _apply_modeled_costs_to_trades  # noqa: E402

def test_debug_no_costs_zero():
    cfg = {"debug_no_costs": True}
    trades = pd.DataFrame({
        "symbol": ["RRC"],
        "entry_price": [10.0],
        "exit_price": [10.2],
        "qty": [1000],
        "realized_pnl": [200.0],
    })

    out = _apply_modeled_costs_to_trades(trades, cfg=cfg)
    assert float(out["modeled_cost_total"].iloc[0]) == 0.0
    assert float(out["realized_pnl_after_costs"].iloc[0]) == 200.0
