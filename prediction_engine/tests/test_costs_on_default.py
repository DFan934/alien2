import pandas as pd
from pathlib import Path

from scripts.run_backtest import _apply_modeled_costs_to_trades  # noqa: E402

def test_costs_on_reduce_pnl():
    cfg = {"debug_no_costs": False}
    trades = pd.DataFrame({
        "symbol": ["RRC", "BBY"],
        "entry_price": [10.0, 20.0],
        "exit_price": [10.2, 19.9],
        "qty": [1000, 500],
        "realized_pnl": [200.0, -50.0],  # before costs
        # provide half-spread for one trade, omit for the other (fallback kicks in)
        "half_spread_usd": [0.01, None],   # USD per share
        "adv_frac": [0.05, 0.02],          # 5% and 2% of ADV
    })

    out = _apply_modeled_costs_to_trades(trades, cfg=cfg)
    assert "modeled_cost_total" in out.columns
    # costs are positive and reduce pnl when present
    assert (out["modeled_cost_total"] > 0).all()
    assert (out["realized_pnl_after_costs"] <= out["realized_pnl"]).all()
