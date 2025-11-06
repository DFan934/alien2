# tests/test_step4_5_max_net.py
import pandas as pd
from prediction_engine.portfolio.ledger import PortfolioLedger
from prediction_engine.portfolio.risk import RiskLimits, RiskEngine
from scripts.run_backtest import _enforce_risk_on_decisions  # noqa: E402

def test_max_net_limit():
    ts = pd.to_datetime(["1999-01-05T09:30Z","1999-01-05T09:31Z"])
    bars = pd.DataFrame({
        "timestamp": [ts[0], ts[1]],
        "symbol": ["RRC","RRC"],
        "open": [10.0, 10.0],
        "volume": [50_000, 50_000],
    })
    # First long is fine; second long at next minute should be blocked by net cap
    decisions = pd.DataFrame({
        "timestamp":[ts[0], ts[1]], "symbol":["RRC","RRC"],
        "target_qty":[9_000, 9_000], "horizon_bars":[10,10]
    })
    eq = 100_000.0
    ledger = PortfolioLedger(cash=eq)
    lim = RiskLimits(max_gross=1e12, max_net=eq*0.8, per_symbol_cap=1e12, sector_cap=1e12, max_concurrent=10, daily_stop=-1e12)
    kept = _enforce_risk_on_decisions(decisions, bars, RiskEngine(lim), ledger, log=None)
    assert len(kept) == 1
