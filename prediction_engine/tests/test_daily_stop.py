# tests/test_step4_5_daily_stop.py
import pandas as pd
from prediction_engine.portfolio.ledger import PortfolioLedger, TradeFill
from prediction_engine.portfolio.risk import RiskLimits, RiskEngine

def test_daily_stop_blocks_new_entries():
    equity0 = 100_000.0
    ledger = PortfolioLedger(cash=equity0)
    limits = RiskLimits(max_gross=equity0, max_net=equity0,
                        per_symbol_cap=equity0, sector_cap=10*equity0,
                        max_concurrent=10, daily_stop=-100.0)
    risk = RiskEngine(limits)

    # Simulate a losing round-trip first
    entry = TradeFill(symbol="RRC", ts=pd.Timestamp("1999-01-05T10:00Z"), side=+1, qty=1000, price=10.0, fees=0.0)
    exitf = TradeFill(symbol="RRC", ts=pd.Timestamp("1999-01-05T10:10Z"), side=-1, qty=1000, price=9.8, fees=0.0)
    ledger.on_fill(entry); ledger.on_fill(exitf)
    assert ledger.day_pnl < -100.0  # crossed the stop

    # Now a new open should be blocked
    ok, reason = risk.can_open(symbol="BBY", sector=None, qty=1000, price=20.0,
                               now=pd.Timestamp("1999-01-05T10:11Z"),
                               ledger_snapshot={**ledger.snapshot_row(), "positions": [], "symbol_notional": {}},
                               open_positions=0)
    assert ok is False and reason == "daily_stop"
