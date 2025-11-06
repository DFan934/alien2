import pandas as pd
from prediction_engine.portfolio.sizer import size_from_p, RiskCaps

def test_ramp_and_caps_nontrivial():
    caps = RiskCaps(max_gross_frac=0.10, adv_cap_pct=0.20)
    capital = 100_000
    price = 20.0
    vol = 0.01  # 1% per-bar sigma
    base_costs = {"price": price, "half_spread_usd": 0.01, "commission": 0.005, "slippage_bp": 0.0}

    q_low  = size_from_p(0.53, vol=vol, capital=capital, risk_caps=caps, costs=base_costs, p_gate=0.55, p_full=0.65)
    q_mid  = size_from_p(0.60, vol=vol, capital=capital, risk_caps=caps, costs=base_costs, p_gate=0.55, p_full=0.65)
    q_high = size_from_p(0.80, vol=vol, capital=capital, risk_caps=caps, costs=base_costs, p_gate=0.55, p_full=0.65)

    assert q_low == 0.0
    assert abs(q_mid) > 0 and abs(q_high) > abs(q_mid)  # monotone with p
