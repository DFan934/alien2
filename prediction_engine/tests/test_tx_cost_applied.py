# tests/test_tx_cost_applied.py
import types
from execution.risk_manager import RiskManager
from prediction_engine.tx_cost import BasicCostModel

def test_roundtrip_deducts_cost():
    rm = RiskManager(account_equity=10_000.0, cost_model=BasicCostModel())
    # Buy 100 @ $10, sell 100 @ $10 → PnL=0 before costs
    is_closed, pnl, _ = rm.process_fill({"price":10.0,"size":100,"side":"buy","trade_id":"t"})
    assert not is_closed and pnl == 0
    eq_before = rm.account_equity
    is_closed, pnl, _ = rm.process_fill({"price":10.0,"size":100,"side":"sell","trade_id":"t"})
    assert is_closed and pnl == 0
    # Equity must be lower by > 0 due to per-fill cost
    assert rm.account_equity < eq_before
