from prediction_engine.portfolio.sizer import size_from_p, RiskCaps

def test_cost_hurdle_blocks_when_edge_too_small():
    caps = RiskCaps(max_gross_frac=0.10, adv_cap_pct=0.20)
    capital = 100_000
    # Make costs big: wide spread + commission
    costs = {"price": 10.0, "half_spread_usd": 0.05, "commission": 0.01, "slippage_bp": 0.0}
    # Small vol and marginal p should be blocked
    qty = size_from_p(0.56, vol=0.004, capital=capital, risk_caps=caps, costs=costs, p_gate=0.55, p_full=0.65, cost_lambda=1.2)
    assert qty == 0.0
