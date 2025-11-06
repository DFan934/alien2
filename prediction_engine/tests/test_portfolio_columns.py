# tests/test_step4_5_portfolio_columns.py
import pandas as pd
from scripts.run_backtest import _apply_modeled_costs_to_trades  # noqa: E402
from prediction_engine.portfolio.ledger import PortfolioLedger, TradeFill
from prediction_engine.portfolio.risk import RiskLimits, RiskEngine

def test_trades_are_augmented_with_portfolio_columns():
    trades = pd.DataFrame({
        "symbol": ["RRC"],
        "entry_ts": [pd.Timestamp("1999-01-05T09:31Z")],
        "entry_price": [10.0],
        "exit_ts": [pd.Timestamp("1999-01-05T09:51Z")],
        "exit_price": [10.2],
        "qty": [1000.0],
        "realized_pnl": [200.0],
        "modeled_cost_total": [5.0],
    })
    cfg = {"equity": 100_000.0, "commission": 0.0, "debug_no_costs": False}
    trades = _apply_modeled_costs_to_trades(trades, cfg)

    # Minimal pass through ledger to add snapshot (simulate what run_backtest does)
    ledger = PortfolioLedger(cash=cfg["equity"])
    for _, r in trades.iterrows():
        ledger.on_fill(TradeFill(symbol=r["symbol"], ts=r["entry_ts"], side=+1, qty=r["qty"], price=r["entry_price"], fees=r["modeled_cost_total"]/2))
        ledger.on_fill(TradeFill(symbol=r["symbol"], ts=r["exit_ts"], side=-1, qty=r["qty"], price=r["exit_price"],  fees=r["modeled_cost_total"]/2))
    snap = ledger.snapshot_row()
    assert {"cash","gross","net","realized_pnl","day_pnl"}.issubset(set(snap.keys()))
