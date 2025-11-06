import pandas as pd
from prediction_engine.portfolio.order_sim import simulate_entry, QuoteStats

def test_partial_fill_ratio_bounds():
    rules = {"max_participation": 0.25, "moo_gap_band": True}
    q = QuoteStats(half_spread_usd=0.01, adv_shares=1_000_000)

    # Case A: volume allows (target 10k; cap = 0.25 * 60k = 15k) → ~100% fill
    entA = simulate_entry(
        decision_row=pd.Series({"target_qty": 10_000, "bar_volume": 60_000}),
        next_open_price=10.0,
        next_open_ts=pd.Timestamp("1999-01-05T09:31Z"),
        quote=q, rules=rules,
    )
    assert entA.fill_ratio >= 0.95

    # Case B: volume capped (target 50k; cap = 15k) → ≤ 0.5 fill
    entB = simulate_entry(
        decision_row=pd.Series({"target_qty": 50_000, "bar_volume": 60_000}),
        next_open_price=10.0,
        next_open_ts=pd.Timestamp("1999-01-05T09:31Z"),
        quote=q, rules=rules,
    )
    assert entB.fill_ratio <= 0.50
