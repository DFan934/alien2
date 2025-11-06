# tests/test_step4_5_sector_cap.py
import pandas as pd
from prediction_engine.portfolio.ledger import PortfolioLedger
from prediction_engine.portfolio.risk import RiskLimits, RiskEngine
from scripts.run_backtest import _enforce_risk_on_decisions  # noqa: E402

def test_sector_cap():
    ts = pd.date_range("1999-01-05 09:30", periods=1, freq="T", tz="UTC")
    bars = pd.DataFrame({
        "timestamp": [ts[0], ts[0]],
        "symbol": ["RRC", "BBY"],
        "open":   [10.0, 20.0],
        "volume": [50_000, 50_000],
    })
    decisions = pd.DataFrame({
        "timestamp": [ts[0], ts[0]],
        "symbol": ["RRC", "BBY"],
        "target_qty": [9_000, 9_000],
        "horizon_bars": [2, 2],
    })
    eq = 100_000.0
    ledger = PortfolioLedger(cash=eq)
    lim = RiskLimits(max_gross=1e12, max_net=1e12, per_symbol_cap=1e12, sector_cap=eq*0.5, max_concurrent=10, daily_stop=-1e12)
    kept = _enforce_risk_on_decisions(
        decisions, bars, RiskEngine(lim), ledger,
        symbol_sector={"RRC":"Energy","BBY":"Energy"}, log=None
    )
    # Each notional: RRC≈$90k; BBY≈$180k; combined sector≈$270k > $50k ⇒ only one should remain
    assert len(kept) == 1
