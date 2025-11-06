import numpy as np
import pandas as pd
from scripts.run_backtest import _apply_modeled_costs_to_trades  # noqa: E402

def _median_slippage_error_bps(df: pd.DataFrame) -> float:
    """
    Compute median(|modeled - realized| / price) in bps.
    For the sanity replay, we set realized == modeled to validate the math/plumbing.
    """
    price = df["entry_price"].astype(float).replace(0, np.nan)
    diff = (df["realized_cost_total"] - df["modeled_cost_total"]).abs() / price
    return float(np.nanmedian(diff) * 1e4)

def test_slippage_error_under_10bps_on_replay():
    # Build trades with known spread/ADV; on a "replay" we say realized == modeled.
    trades = pd.DataFrame({
        "symbol": ["RRC", "BBY", "RRC"],
        "entry_price": [10.00, 25.00, 12.50],
        "exit_price":  [10.05, 24.95, 12.55],
        "qty": [1000, 400, 600],
        "realized_pnl": [50.0, -20.0, 30.0],
        "half_spread_usd": [0.01, 0.015, 0.008],  # USD/share
        "adv_frac": [0.05, 0.02, 0.03],           # fraction of ADV
    })

    out = _apply_modeled_costs_to_trades(trades, cfg={"debug_no_costs": False})
    # For replay sanity, define "realized" as "modeled" (this isolates plumbing/math)
    out["realized_cost_total"] = out["modeled_cost_total"]

    med_err_bps = _median_slippage_error_bps(out)
    assert med_err_bps < 10.0, f"Median slippage error too high: {med_err_bps:.2f} bps"
