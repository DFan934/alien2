from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import pytest

from execution.brokers.base import MockBrokerAdapter
from execution.contracts_live import OrderUpdateEvent, FillUpdate
from execution.manager import ExecutionManager


@pytest.mark.asyncio
async def test_step9_partial_and_full_fill_updates_artifacts(tmp_path: Path):
    mgr = ExecutionManager(equity=10_000)
    broker = MockBrokerAdapter()
    await broker.connect()
    mgr.attach_broker(broker, out_dir=tmp_path)

    now = datetime.now(timezone.utc)

    # ACCEPTED (0 filled)
    accepted = OrderUpdateEvent(
        broker_order_id="mock_1",
        client_order_id="c1",
        symbol="AAPL",
        status="ACCEPTED",
        side="BUY",
        filled_qty=0,
        avg_fill_price=None,
        last_fill=None,
        event_ts_utc=now,
        raw={"mock": True, "stage": "accepted"},
    )
    await broker.push_order_update(accepted)

    # PARTIAL fill: filled_qty 1
    partial = OrderUpdateEvent(
        broker_order_id="mock_1",
        client_order_id="c1",
        symbol="AAPL",
        status="PARTIALLY_FILLED",
        side="BUY",
        filled_qty=1,
        avg_fill_price=100.0,
        #last_fill=FillUpdate(fill_id="f1", qty=1, price=100.0, liquidity=None),
        last_fill=FillUpdate(ts_utc=now, fill_id="f1", qty=1, price=100.0, liquidity=None),
        event_ts_utc=now,
        raw={"mock": True, "stage": "partial"},
    )
    await broker.push_order_update(partial)

    # FULL fill: filled_qty 2 (delta 1)
    full = OrderUpdateEvent(
        broker_order_id="mock_1",
        client_order_id="c1",
        symbol="AAPL",
        status="FILLED",
        side="BUY",
        filled_qty=2,
        avg_fill_price=100.5,
        #last_fill=FillUpdate(fill_id="f2", qty=1, price=101.0, liquidity=None),
        last_fill=FillUpdate(ts_utc=now, fill_id="f2", qty=1, price=101.0, liquidity=None),
        event_ts_utc=now,
        raw={"mock": True, "stage": "full"},
    )
    await broker.push_order_update(full)

    # Let consumer run
    await asyncio.sleep(0.2)

    # Consumer sanity: ensure we actually processed updates
    assert mgr._order_updates_consumer_exc is None, f"consumer crashed: {mgr._order_updates_consumer_exc}"
    assert getattr(mgr, "_ou_seen", 0) >= 1, "consumer did not see any order updates"


    # Artifacts exist
    assert any((tmp_path / "fills.parquet").glob("part-*.parquet"))
    assert any((tmp_path / "positions.parquet").glob("part-*.parquet"))

    # State updated (2 shares long)
    assert float(getattr(mgr.risk_mgr, "position_size", 0.0)) == pytest.approx(2.0)

    await broker.close()
