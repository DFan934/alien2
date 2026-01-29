import asyncio
import pytest
from pathlib import Path

from execution.manager import ExecutionManager
from execution.brokers.base import MockBrokerAdapter
from execution.exit_manager import ExitManager

@pytest.mark.asyncio
async def test_step10_tp1_then_stop_submits_exit_orders(tmp_path: Path):
    mgr = ExecutionManager(equity=10_000)

    entry_px = 100.0
    tp2 = entry_px * (1.0 + ExitManager.TP2_PCT)

    mgr.store.add_position(
        signal_id="s1",
        symbol="AAPL",
        side="BUY",
        qty=2,
        entry_px=entry_px,
        stop_px=98.0,
        tp_remaining=tp2,
    )

    broker = MockBrokerAdapter()
    await broker.connect()
    mgr.attach_broker(broker, out_dir=tmp_path)

    try:
        feats = {"ema_fast_dist": 0.0, "vwap_dist": 0.0, "atr": 0.0, "orderflow_delta": 0.0}
        # TP1 should submit a sell for 1
        await mgr._process_stop_and_exit("AAPL", 102.0, feats)

        # STOP should submit a sell for remaining 1
        await mgr._process_stop_and_exit("AAPL", 97.0, feats)

        # Assert broker got at least 2 orders
        orders = getattr(broker, "_orders", None)
        assert orders is not None, "MockBrokerAdapter has no _orders field—check mock implementation"
        assert len(orders) >= 2, f"expected >=2 orders, got {len(orders)}"

    finally:
        # Always run cleanup even if asserts fail
        await mgr.aclose()
        await broker.close()
