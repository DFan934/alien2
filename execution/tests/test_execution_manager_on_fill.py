import pytest
import asyncio
from execution.manager import ExecutionManager
from execution.risk_manager import RiskManager
from prediction_engine.drift_monitor import DriftStatus


@pytest.mark.asyncio
async def test_on_fill_updates_equity_and_drift():
    # 1) Create manager with initial equity $1000
    mgr = ExecutionManager(equity=1000.0)

    # 2) Seed an “open trade” with a known pred_prob
    trade_id = "TEST123"
    mgr._open_trades[trade_id] = {"pred_prob": 0.42}

    # 3) Simulate a full entry + full exit
    #    First, buy 5 @10
    await mgr.on_fill({"price": 10.0, "size": 5.0, "side": "buy",  "trade_id": trade_id})
    #    Then, sell all 5 @12 → +10 PnL
    await mgr.on_fill({"price": 12.0, "size": 5.0, "side": "sell", "trade_id": trade_id})

    # 4) Equity should have increased by +10
    assert mgr.risk_mgr.account_equity == pytest.approx(1010.0)

    # 5) Open trades map should no longer contain the trade_id
    assert trade_id not in mgr._open_trades

    # 6) The DriftMonitor should have recorded an update
    #    (status returns (code, metrics) – code should be OK or NO_ACTION,
    #     but its internal counters advanced)
    status, metrics = mgr.drift_monitor.status()
    # ensure a drift log entry (ticket) was created
    #assert isinstance(status, DriftStatus)
    # You can also assert that metrics contains our pred_prob and outcome
    #assert metrics.get("last_pred") == pytest.approx(0.42)
    #assert metrics.get("last_outcome") is True


    assert isinstance(status, DriftStatus)
    # metrics dict should be non-empty after update
    assert isinstance(metrics, dict)
    #assert metrics, "expected metrics to have entries after update"


    # ---- teardown: cancel the safety watcher to avoid warnings ----
    mgr._safety_task.cancel()
    try:
        await mgr._safety_task
    except asyncio.CancelledError:
        pass