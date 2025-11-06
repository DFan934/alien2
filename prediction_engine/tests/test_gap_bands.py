import pandas as pd
from prediction_engine.portfolio.order_sim import simulate_entry, simulate_exit, QuoteStats

def test_moo_moc_gap_bands_apply_half_spread():
    q = QuoteStats(half_spread_usd=0.02, adv_shares=1_000_000)
    rules = {"moo_gap_band": True, "moc_gap_band": True}

    # Long entry → entry price = open + half_spread
    ent = simulate_entry(
        decision_row=pd.Series({"target_qty": 1000, "bar_volume": 100000}),
        next_open_price=20.00,
        next_open_ts=pd.Timestamp("1999-01-05T09:31Z"),
        quote=q, rules=rules,
    )
    assert abs(ent.entry_price - 20.02) < 1e-9

    # Long exit → exit price = open - half_spread
    ex = simulate_exit(
        position_row=pd.Series({"filled_qty": ent.filled_qty, "entry_price": ent.entry_price}),
        exit_open_price=21.00,
        exit_open_ts=pd.Timestamp("1999-01-05T10:31Z"),
        bars_held=20,
        quote=q, rules=rules,
    )
    assert abs(ex.exit_price - 20.98) < 1e-9
