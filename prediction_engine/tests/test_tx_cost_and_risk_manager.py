from prediction_engine.tx_cost import BasicCostModel
from execution.risk_manager import RiskManager

def test_costs_reduce_equity_on_round_trip():
    # No-cost account: round-trip at same price => equity unchanged
    rm_free = RiskManager(account_equity=10_000.0, cost_model=None)
    rm_free.process_fill({"price": 10.0, "size": 100, "side": "buy"})
    is_closed, pnl, _ = rm_free.process_fill({"price": 10.0, "size": 100, "side": "sell"})
    assert is_closed and abs(pnl) < 1e-9
    eq_free = rm_free.account_equity

    # With costs: same round-trip => equity strictly lower
    rm_cost = RiskManager(account_equity=10_000.0, cost_model=BasicCostModel())
    rm_cost.process_fill({"price": 10.0, "size": 100, "side": "buy"})
    is_closed2, pnl2, _ = rm_cost.process_fill({"price": 10.0, "size": 100, "side": "sell"})
    assert is_closed2 and abs(pnl2) < 1e-9
    eq_cost = rm_cost.account_equity

    assert eq_cost < eq_free, "Equity with costs must be lower than without costs"
    assert (eq_free - eq_cost) > 0.0
