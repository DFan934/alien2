# tests/test_step4_5_max_concurrent.py
import pandas as pd
from prediction_engine.portfolio.ledger import PortfolioLedger
from prediction_engine.portfolio.risk import RiskLimits, RiskEngine
from scripts.run_backtest import _enforce_risk_on_decisions  # noqa: E402

def test_max_concurrent_enforced(caplog):
    ts = pd.date_range("1999-01-05 09:30", periods=3, freq="T", tz="UTC")
    bars = pd.DataFrame({
        "timestamp": list(ts)*2,
        "symbol": ["RRC"]*3 + ["BBY"]*3,
        "open":   [10,10,10] + [20,20,20],
        "volume": [50_000]*6,
    })
    decisions = pd.DataFrame({
        "timestamp": [ts[0], ts[0], ts[1]],
        "symbol": ["RRC", "BBY", "RRC"],
        "target_qty": [10_000, 10_000, 10_000],
        "horizon_bars": [2,2,2],
    })

    equity0 = 100_000.0
    ledger = PortfolioLedger(cash=equity0)
    limits = RiskLimits(
        max_gross=equity0 * 10.0,  # big
        max_net=equity0 * 10.0,  # big
        per_symbol_cap=equity0 * 10.0,
        sector_cap=equity0 * 10.0,
        max_concurrent=1,
        daily_stop=-equity0 * 0.5
    )

    risk = RiskEngine(limits)

    kept = _enforce_risk_on_decisions(decisions, bars, risk, ledger, log=None)
    # At t0 only one of the two entries should survive; the t1 entry can proceed
    assert len(kept) == 2
