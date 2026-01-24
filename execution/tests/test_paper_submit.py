from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from execution.brokers.base import MockBrokerAdapter
from execution.contracts_live import OrderRequest
from execution.manager import ExecutionManager


@pytest.mark.asyncio
async def test_step8_tiny_qty_and_daily_limit(tmp_path: Path):
    mgr = ExecutionManager(equity=10_000)  # uses lightweight init path
    broker = MockBrokerAdapter()
    await broker.connect()

    mgr.attach_broker(broker, out_dir=tmp_path)

    # Force tiny sizing + 1/day
    mgr._tiny_qty = 1
    mgr._order_limit_per_day = 1

    # 1) First order allowed, qty forced to 1
    req1 = OrderRequest(symbol="AAPL", side="BUY", qty=10)
    evt1 = await mgr.submit_order_paper(req1)
    assert evt1 is not None
    assert evt1.client_order_id == req1.client_order_id

    # 2) Second order same day blocked
    req2 = OrderRequest(symbol="AAPL", side="BUY", qty=10)
    evt2 = await mgr.submit_order_paper(req2)
    assert evt2 is None

    # Verify datasets exist
    assert any((tmp_path / "attempted_actions.parquet").glob("part-*.parquet"))
    assert any((tmp_path / "orders.parquet").glob("part-*.parquet"))

    await broker.close()
