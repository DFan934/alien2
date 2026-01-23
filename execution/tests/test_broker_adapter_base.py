from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from data_ingestion.contracts import LiveBar, MarketDataEvent
from execution.brokers.base import MockBrokerAdapter
from execution.contracts_live import OrderRequest


@pytest.mark.asyncio
async def test_mock_adapter_stream_market_data_filters_and_yields():
    broker = MockBrokerAdapter()
    await broker.connect()

    t0 = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    bar = LiveBar(
        symbol="AAPL",
        ts_start_utc=t0,
        ts_end_utc=t0 + timedelta(minutes=1),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=10.0,
        source="test",
        is_final=True,
    )

    evt_aapl = MarketDataEvent(event_type="BAR", symbol="AAPL", event_ts_utc=t0, bar=bar)
    evt_msft = MarketDataEvent(event_type="BAR", symbol="MSFT", event_ts_utc=t0, bar=bar.copy(update={"symbol": "MSFT"}))

    # Start consumer
    got = []

    async def consume_one():
        async for e in broker.stream_market_data(symbols=["AAPL"]):
            got.append(e)
            break

    task = asyncio.create_task(consume_one())

    # Push events (MSFT should be filtered out)
    await broker.push_market_event(evt_msft)
    await broker.push_market_event(evt_aapl)

    await asyncio.wait_for(task, timeout=2.0)

    assert len(got) == 1
    assert got[0].symbol == "AAPL"
    assert got[0].event_type == "BAR"
    assert got[0].bar is not None
    assert got[0].bar.symbol == "AAPL"

    await broker.close()


@pytest.mark.asyncio
async def test_mock_adapter_submit_order_accepts():
    broker = MockBrokerAdapter()
    await broker.connect()

    req = OrderRequest(
        symbol="AAPL",
        side="BUY",
        qty=1,
        order_type="MKT",
        tif="DAY",
        reason="unit test",
    )

    upd = await broker.submit_order(req)

    assert upd.client_order_id == req.client_order_id
    assert upd.symbol == "AAPL"
    assert upd.status == "ACCEPTED"
    assert upd.side == "BUY"
    assert upd.broker_order_id is not None

    await broker.close()
