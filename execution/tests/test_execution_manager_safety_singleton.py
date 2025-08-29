import pytest
from execution.manager import ExecutionManager

@pytest.mark.asyncio
async def test_singleton_safety_instance_and_state_persists():
    mgr = ExecutionManager(equity=100_000.0, config={"safety": {"latency_ms": 1.0}})
    sid0 = id(mgr.safety)

    # Call on_bar a few times; instance must not change
    for i in range(5):
        await mgr.on_bar({"ts": i, "symbol": "TEST", "price": 100.0, "features": {}})
    assert id(mgr.safety) == sid0

    # Force a halt via latency metric; instance must still be identical
    assert mgr.safety.should_halt(latency_ms=10_000.0) is True
    assert id(mgr.safety) == sid0
