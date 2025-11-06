# tests/test_step4_5_per_symbol_cap.py
import pandas as pd
from prediction_engine.portfolio.ledger import PortfolioLedger
from prediction_engine.portfolio.risk import RiskLimits, RiskEngine
from scripts.run_backtest import _enforce_risk_on_decisions  # noqa: E402

def test_per_symbol_cap_blocks():
    ts = pd.to_datetime(["1999-01-05T09:30Z"])
    bars = pd.DataFrame({"timestamp":[ts[0]], "symbol":["RRC"], "open":[10.0], "volume":[50_000]})
    decisions = pd.DataFrame({"timestamp":[ts[0]], "symbol":["RRC"], "target_qty":[20_000], "horizon_bars":[2]})
    eq = 100_000.0
    ledger = PortfolioLedger(cash=eq)
    lim = RiskLimits(max_gross=1e12, max_net=1e12, per_symbol_cap=eq*0.1, sector_cap=1e12, max_concurrent=10, daily_stop=-1e12)
    kept = _enforce_risk_on_decisions(decisions, bars, RiskEngine(lim), ledger, log=None)
    # 20k * $10 = $200k > per_symbol_cap ($10k) ⇒ blocked
    assert kept.empty
