import pytest
from execution.latency import latency_monitor
from execution.manager import ExecutionManager

@pytest.mark.asyncio
async def test_timeit_decorator_records_every_call():
    mgr = ExecutionManager(equity=100_000.0)
    latency_monitor.reset("execution_bar")

    await mgr.on_bar({"ts": 1, "symbol": "X", "price": 10.0, "features": {}})
    await mgr.on_bar({"ts": 2, "symbol": "X", "price": 10.0, "features": {}})

    # 2 calls → 2 samples recorded (since @timeit wraps the async function)
    assert latency_monitor.count("execution_bar") == 2
