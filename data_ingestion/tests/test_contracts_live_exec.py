# execution/tests/test_contracts_live_exec.py

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from execution.contracts_live import FillUpdate, OrderRequest, OrderUpdateEvent


def test_orderrequest_side_normalizes():
    """
    OrderRequest(side="buy", ...) -> assert side == "BUY"
    """
    req = OrderRequest(
        symbol="AAPL",
        side="buy",
        qty=1,
        order_type="MKT",
        tif="DAY",
        reason="test",
    )
    assert req.side == "BUY"


def test_orderupdate_rejects_naive_event_ts():
    """
    OrderUpdateEvent(event_ts_utc=datetime.utcnow(), ...) -> expect validation failure
    because event_ts_utc must be timezone-aware.
    """
    naive_ts = datetime.utcnow()

    with pytest.raises((ValueError, ValidationError)):
        OrderUpdateEvent(
            client_order_id="cid_test_1",
            broker_order_id="bid_test_1",
            symbol="AAPL",
            status="ACCEPTED",
            side="BUY",
            filled_qty=0,
            event_ts_utc=naive_ts,  # naive (should fail)
            raw={"test": True},
        )


def test_fillupdate_roundtrip():
    """
    Create FillUpdate tz-aware; embed into OrderUpdateEvent(last_fill=...);
    round-trip parse.
    """
    t_fill = datetime.now(timezone.utc).replace(microsecond=0)
    fill = FillUpdate(qty=1, price=100.25, ts_utc=t_fill)

    upd = OrderUpdateEvent(
        client_order_id="cid_test_2",
        broker_order_id="bid_test_2",
        symbol="AAPL",
        status="PARTIALLY_FILLED",
        side="SELL",
        filled_qty=1,
        avg_fill_price=100.25,
        last_fill=fill,
        event_ts_utc=t_fill + timedelta(seconds=1),
        raw={"broker_payload": "x"},
    )

    upd2 = OrderUpdateEvent.parse_obj(upd.dict())
    assert upd2.client_order_id == upd.client_order_id
    assert upd2.broker_order_id == upd.broker_order_id
    assert upd2.status == upd.status
    assert upd2.last_fill is not None
    assert upd2.last_fill.qty == 1
    assert float(upd2.last_fill.price) == 100.25


from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from execution.contracts_live import OrderRequest, OrderUpdateEvent


def test_orderrequest_limit_requires_limit_price():
    """
    If order_type == "LMT", limit_price must be present.
    """
    with pytest.raises((ValueError, ValidationError)):
        OrderRequest(
            symbol="AAPL",
            side="BUY",
            qty=1,
            order_type="LMT",
            tif="DAY",
            limit_price=None,  # invalid
        )


def test_orderrequest_stop_requires_stop_price():
    """
    If order_type == "STP", stop_price must be present.
    """
    with pytest.raises((ValueError, ValidationError)):
        OrderRequest(
            symbol="AAPL",
            side="BUY",
            qty=1,
            order_type="STP",
            tif="DAY",
            stop_price=None,  # invalid
        )


def test_orderupdate_status_enum_rejects_unknown():
    """
    Unknown status should be rejected (ensures status space is frozen).
    """
    t0 = datetime.now(timezone.utc)
    with pytest.raises((ValueError, ValidationError)):
        OrderUpdateEvent(
            client_order_id="cid_test_status",
            broker_order_id="bid_test_status",
            symbol="AAPL",
            status="WUT",  # invalid
            side="BUY",
            filled_qty=0,
            event_ts_utc=t0,
            raw={"test": True},
        )
