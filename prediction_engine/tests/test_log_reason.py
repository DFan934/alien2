# tests/test_step4_5_log_reason.py
import pandas as pd
import logging
from prediction_engine.portfolio.ledger import PortfolioLedger
from prediction_engine.portfolio.risk import RiskLimits, RiskEngine
from scripts.run_backtest import _enforce_risk_on_decisions  # noqa: E402

def test_log_reason_emitted(caplog):
    ts = pd.date_range("1999-01-05 09:30", periods=1, freq="T", tz="UTC")
    bars = pd.DataFrame({
        "timestamp": [ts[0]],
        "symbol": ["RRC"],
        "open":   [10.0],
        "volume": [50_000],
    })
    decisions = pd.DataFrame({
        "timestamp": [ts[0]],
        "symbol": ["RRC"],
        "target_qty": [1_000_000],  # huge to trip per_symbol_cap
        "horizon_bars": [2],
    })
    equity0 = 100_000.0
    ledger = PortfolioLedger(cash=equity0)
    limits = RiskLimits(
        max_gross=1e12, max_net=1e12,
        per_symbol_cap=equity0 * 0.05,  # tight
        sector_cap=1e12, max_concurrent=10, daily_stop=-1e12
    )
    risk = RiskEngine(limits)
    logger = logging.getLogger("risk_test")
    with caplog.at_level(logging.INFO, logger="risk_test"):
        kept = _enforce_risk_on_decisions(decisions, bars, risk, ledger, log=logger)
    assert kept.empty
    assert any("per_symbol_cap" in rec.message for rec in caplog.records)
