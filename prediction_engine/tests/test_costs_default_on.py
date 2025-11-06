import pandas as pd
from scripts.run_backtest import _apply_modeled_costs_to_trades  # noqa: E402

def test_costs_default_on_when_cfg_missing():
    cfg = {}  # simulate default config path
    trades = pd.DataFrame({
        "symbol": ["RRC"],
        "entry_price": [10.0],
        "exit_price": [10.1],
        "qty": [1000],
        "realized_pnl": [100.0],
        # omit half_spread/adv to exercise model fallbacks
    })
    out = _apply_modeled_costs_to_trades(trades, cfg=cfg)
    assert "modeled_cost_total" in out.columns
    assert float(out["modeled_cost_total"].iloc[0]) > 0.0
    assert float(out["realized_pnl_after_costs"].iloc[0]) < 100.0
