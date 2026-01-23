# data_ingestion/tests/test_contracts_live_ingest.py

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from data_ingestion.contracts import LiveBar, MarketDataEvent


def test_livebar_rejects_naive_datetime():
    """
    Build LiveBar(ts_start_utc=datetime.utcnow(), ...) -> expect validation failure
    because ts_start_utc must be timezone-aware.
    """
    naive_start = datetime.utcnow()
    aware_end = datetime.now(timezone.utc) + timedelta(minutes=1)

    with pytest.raises((ValueError, ValidationError)):
        LiveBar(
            symbol="AAPL",
            ts_start_utc=naive_start,      # naive (should fail)
            ts_end_utc=aware_end,
            open=100.0,
            high=101.0,
            low=99.5,
            close=100.5,
            volume=1234.0,
            source="test",
            is_final=True,
        )


def test_marketdataevent_bar_payload_roundtrip():
    """
    Build LiveBar with tz-aware times; embed in MarketDataEvent(event_type="BAR", ...).
    Assert event.bar.symbol == event.symbol.
    Assert MarketDataEvent.parse_obj(event.dict()) succeeds.
    """
    t0 = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    t1 = t0 + timedelta(minutes=1)

    bar = LiveBar(
        symbol="AAPL",
        ts_start_utc=t0,
        ts_end_utc=t1,
        open=100.0,
        high=101.0,
        low=99.5,
        close=100.5,
        volume=1234.0,
        source="test",
        is_final=True,
    )

    evt = MarketDataEvent(
        event_type="BAR",
        symbol="AAPL",
        event_ts_utc=t1,
        bar=bar,
    )

    assert evt.bar is not None
    assert evt.bar.symbol == evt.symbol

    # round-trip: dict -> parse_obj
    evt2 = MarketDataEvent.parse_obj(evt.dict())
    assert evt2.symbol == evt.symbol
    assert evt2.event_type == evt.event_type
    assert evt2.bar is not None
    assert evt2.bar.symbol == "AAPL"



from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from data_ingestion.contracts import LiveBar, MarketDataEvent


def test_livebar_requires_ts_end_after_start():
    """
    ts_end_utc must be strictly after ts_start_utc.
    """
    t0 = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    with pytest.raises((ValueError, ValidationError)):
        LiveBar(
            symbol="AAPL",
            ts_start_utc=t0,
            ts_end_utc=t0,  # invalid: not after start
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000.0,
            source="test",
            is_final=True,
        )

    with pytest.raises((ValueError, ValidationError)):
        LiveBar(
            symbol="AAPL",
            ts_start_utc=t0,
            ts_end_utc=t0 - timedelta(seconds=1),  # invalid: before start
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.5,
            volume=1000.0,
            source="test",
            is_final=True,
        )


def test_livebar_ohlc_invariants():
    """
    Enforce basic OHLC sanity:
      - high >= max(open, close, low)
      - low <= min(open, close, high)
      - low <= high
    """
    t0 = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    t1 = t0 + timedelta(minutes=1)

    # invalid: high < open
    with pytest.raises((ValueError, ValidationError)):
        LiveBar(
            symbol="AAPL",
            ts_start_utc=t0,
            ts_end_utc=t1,
            open=100.0,
            high=99.0,  # invalid
            low=98.0,
            close=99.5,
            volume=1000.0,
            source="test",
            is_final=True,
        )

    # invalid: low > close
    with pytest.raises((ValueError, ValidationError)):
        LiveBar(
            symbol="AAPL",
            ts_start_utc=t0,
            ts_end_utc=t1,
            open=100.0,
            high=102.0,
            low=101.0,  # invalid if close < low
            close=100.5,
            volume=1000.0,
            source="test",
            is_final=True,
        )

    # invalid: low > high
    with pytest.raises((ValueError, ValidationError)):
        LiveBar(
            symbol="AAPL",
            ts_start_utc=t0,
            ts_end_utc=t1,
            open=100.0,
            high=100.0,
            low=101.0,  # invalid
            close=100.5,
            volume=1000.0,
            source="test",
            is_final=True,
        )


def test_marketdataevent_bar_type_requires_bar_payload():
    """
    If event_type == "BAR", then bar payload must be present (not None).
    """
    t0 = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    with pytest.raises((ValueError, ValidationError)):
        MarketDataEvent(
            event_type="BAR",
            symbol="AAPL",
            event_ts_utc=t0,
            bar=None,  # invalid for BAR
        )
