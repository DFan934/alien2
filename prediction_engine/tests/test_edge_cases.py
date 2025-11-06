import pandas as pd
from prediction_engine.portfolio.order_sim import simulate_entry, QuoteStats

def test_zero_qty_or_volume():
    q = QuoteStats(half_spread_usd=0.01, adv_shares=1_000_000)
    rules = {"max_participation": 0.25}

    # target = 0 → no fill
    ent0 = simulate_entry(
        decision_row=pd.Series({"target_qty": 0, "bar_volume": 50_000}),
        next_open_price=10.0,
        next_open_ts=pd.Timestamp("1999-01-05T09:31Z"),
        quote=q, rules=rules,
    )
    assert ent0.filled_qty == 0 and ent0.fill_ratio == 0

    # zero bar volume → cap is 0 → fill ratio 0
    entZ = simulate_entry(
        decision_row=pd.Series({"target_qty": 10_000, "bar_volume": 0}),
        next_open_price=10.0,
        next_open_ts=pd.Timestamp("1999-01-05T09:31Z"),
        quote=q, rules=rules,
    )
    assert entZ.filled_qty == 0 and entZ.fill_ratio == 0
