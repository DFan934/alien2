import asyncio
import math
import pytest

from execution.latency import latency_monitor
from execution.manager import ExecutionManager

@pytest.mark.asyncio
async def test_latency_counts_and_stats_async_on_bar():
    # Use equity=... path to avoid wiring full dependencies
    mgr = ExecutionManager(equity=100_000.0)

    latency_monitor.reset("execution_bar")
    # Call fewer than 60 times so on_bar returns early and never touches EV
    for i in range(10):
        bar = {"ts": i, "symbol": "TEST", "price": 100.0, "features": {}}
        await mgr.on_bar(bar)

    assert latency_monitor.count("execution_bar") == 10
    assert latency_monitor.mean("execution_bar") >= 0.0
    assert math.isfinite(latency_monitor.p95("execution_bar"))
