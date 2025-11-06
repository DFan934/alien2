import pandas as pd
from scripts.run_backtest import _apply_sizer_to_decisions, _simulate_trades_from_decisions  # noqa: E402

def test_sizer_then_simulator_produces_filled_trades():
    ts = pd.date_range("1999-01-05 09:30", periods=12, freq="T", tz="UTC")
    bars = pd.DataFrame({
        "timestamp": list(ts)*2,
        "symbol": ["RRC"]*12 + ["BBY"]*12,
        "open":   [10,10.1,10.2,10.3,10.4,10.5,10.6,10.7,10.8,10.9,11.0,11.1] +
                  [20,20.1,20.2,20.3,20.4,20.5,20.6,20.7,20.8,20.9,21.0,21.1],
        "volume": [60_000]*24,
    })
    decisions = pd.DataFrame({
        "timestamp": [ts[2], ts[3], ts[4], ts[5]],
        "symbol": ["RRC", "BBY", "RRC", "BBY"],
        "p_cal":  [0.52, 0.57, 0.63, 0.72],
        "horizon_bars": [3, 3, 3, 3],
    })

    cfg = {"equity": 100_000.0, "p_gate_quantile": 0.55, "full_p_quantile": 0.65,
           "commission": 0.005, "slippage_bp": 0.0, "impact_bps_per_adv_frac": 25.0,
           "adv_cap_pct": 0.20, "max_gross_frac": 0.10}

    cfg.update({
        "sizer_strategy": "prob",  # or "kelly" if your sizer supports it
        "sizer_cost_lambda": 0.8,  # lower the hurdle so small edges size up
        "spread_bp": 2.0,  # make sure half-spread fallback is sane
    })

    sized = _apply_sizer_to_decisions(decisions, bars, cfg)
    # Below gate p=0.52 should be zero; higher ps should be > 0
    assert float(sized.loc[sized["p_cal"] <= 0.55, "target_qty"].fillna(0).sum()) == 0.0
    assert (sized.loc[sized["p_cal"] > 0.55, "target_qty"].abs() > 0).any()

    trades = _simulate_trades_from_decisions(sized, bars, rules={"max_participation": 0.25, "moo_gap_band": True, "moc_gap_band": True})
    # Non-trivial qty distribution (not mostly zeros)
    assert (trades["qty"].abs() > 0).mean() == 1.0


from prediction_engine.portfolio.sizer import size_from_p, RiskCaps

def test_side_sign_long_vs_short():
    caps = RiskCaps(max_gross_frac=0.1, adv_cap_pct=0.2)
    costs = {"price": 20.0, "half_spread_usd": 0.01, "commission": 0.0}
    long_q  = size_from_p(0.60, vol=0.01, capital=100_000, risk_caps=caps, costs=costs)
    short_q = size_from_p(0.40, vol=0.01, capital=100_000, risk_caps=caps, costs=costs)
    assert long_q > 0.0 and short_q < 0.0




from prediction_engine.portfolio.sizer import size_from_p, RiskCaps

def test_adv_cap_limits_shares():
    caps = RiskCaps(max_gross_frac=1.0, adv_cap_pct=0.10)  # 10% of ADV
    costs = {"price": 10.0, "half_spread_usd": 0.0, "commission": 0.0, "adv_shares": 100_000}
    q = size_from_p(0.80, vol=0.02, capital=10_000_000, risk_caps=caps, costs=costs)
    assert abs(q) <= 0.10 * 100_000 + 1e-6




from prediction_engine.portfolio.sizer import size_from_p, RiskCaps

def test_ramp_continuity_at_p_full():
    caps = RiskCaps()
    costs = {"price": 50.0, "half_spread_usd": 0.01}
    p_gate, p_full = 0.55, 0.65
    q1 = size_from_p(p_full - 1e-6, vol=0.01, capital=1e6, risk_caps=caps, costs=costs,
                     p_gate=p_gate, p_full=p_full)
    q2 = size_from_p(p_full + 1e-6, vol=0.01, capital=1e6, risk_caps=caps, costs=costs,
                     p_gate=p_gate, p_full=p_full)
    assert abs(q2 - q1) / max(1.0, abs(q1)) < 0.05  # ≤5% jump




import math
from prediction_engine.portfolio.sizer import size_from_p, RiskCaps

def test_every_taken_trade_clears_cost_hurdle():
    caps = RiskCaps(max_gross_frac=0.2, adv_cap_pct=1.0)
    price = 25.0
    costs = {"price": price, "half_spread_usd": 0.01, "commission": 0.005, "slippage_bp": 0.0}
    for p in [0.56, 0.60, 0.68, 0.72]:
        q = size_from_p(p, vol=0.01, capital=100_000, risk_caps=caps, costs=costs,
                        p_gate=0.55, p_full=0.65, cost_lambda=1.2)
        if q != 0.0:
            mu = (2*p - 1) * 0.01
            ev_per_share = mu * price
            model_cost = (2*costs["half_spread_usd"] + 2*costs["commission"])  # no slip/impact here
            assert ev_per_share >= 1.2 * model_cost - 1e-9





from prediction_engine.portfolio.sizer import size_from_p, RiskCaps

def test_strategy_alias_prob_acts_like_score():
    caps = RiskCaps()
    costs = {"price": 30.0, "half_spread_usd": 0.01}
    q_score = size_from_p(0.62, vol=0.01, capital=100_000, risk_caps=caps, costs=costs,
                          strategy="score")
    # assuming you added the alias in sizer; otherwise expect a KeyError/ValueError
    q_prob  = size_from_p(0.62, vol=0.01, capital=100_000, risk_caps=caps, costs=costs,
                          strategy="prob")
    assert abs(q_prob - q_score) / max(1.0, abs(q_score)) < 0.05




