# execution/tests/test_alpaca_paper_policy.py
from __future__ import annotations

import asyncio
from datetime import timezone

import pytest
import urllib.request
import urllib.error

from execution.brokers.alpaca_paper import AlpacaPaperAdapter, _parse_rfc3339_to_utc
from execution.brokers.base import BrokerUnavailable


def _mk_cfg(*, max_retries: int, backoff_s: tuple[float, ...], breaker_failures: int, breaker_cooloff_s: float):
    # Provide direct credentials so __init__ doesn't depend on env.
    return {
        "broker": {
            "alpaca": {
                "key_id": "TEST_KEY",
                "secret_key": "TEST_SECRET",
                "trading_base_url": "https://paper-api.alpaca.markets",
                "api_version": "v2",
            },
            "call_policy": {
                "timeout_s": 0.01,
                "max_retries": max_retries,
                "backoff_s": list(backoff_s),
                "circuit_breaker_failures": breaker_failures,
                "circuit_breaker_cooloff_s": breaker_cooloff_s,
            },
        }
    }


def test_parse_rfc3339_to_utc_is_tz_aware():
    dt = _parse_rfc3339_to_utc("2026-01-22T14:30:00Z")
    assert dt.tzinfo is not None
    assert dt.utcoffset() == timezone.utc.utcoffset(dt)


@pytest.mark.asyncio
async def test_alpaca_retries_then_unavailable(monkeypatch):
    """
    If network is down, adapter retries (1 + max_retries) and then raises BrokerUnavailable.
    """
    cfg = _mk_cfg(max_retries=1, backoff_s=(0.0,), breaker_failures=999, breaker_cooloff_s=9999.0)
    broker = AlpacaPaperAdapter(cfg)
    await broker.connect()

    calls = {"n": 0}

    def _fake_urlopen(*args, **kwargs):
        calls["n"] += 1
        raise urllib.error.URLError("no network")

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    with pytest.raises(BrokerUnavailable):
        await broker.get_clock()

    # attempts = 1 + max_retries = 2
    assert calls["n"] == 2

    await broker.close()


@pytest.mark.asyncio
async def test_circuit_breaker_opens_and_fails_fast(monkeypatch):
    """
    After N consecutive failures, the circuit breaker opens.
    Next call should fail fast WITHOUT calling urlopen again.
    """
    cfg = _mk_cfg(max_retries=0, backoff_s=(0.0,), breaker_failures=2, breaker_cooloff_s=9999.0)
    broker = AlpacaPaperAdapter(cfg)
    await broker.connect()

    calls = {"n": 0}

    def _fake_urlopen(*args, **kwargs):
        calls["n"] += 1
        raise urllib.error.URLError("no network")

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    # First call fails: breaker failure count = 1
    with pytest.raises(BrokerUnavailable):
        await broker.get_clock()
    assert calls["n"] == 1

    # Second call fails: breaker opens (threshold=2)
    with pytest.raises(BrokerUnavailable):
        await broker.get_clock()
    assert calls["n"] == 2

    # Third call should fail fast due to open breaker (no extra urlopen call)
    with pytest.raises(BrokerUnavailable) as e:
        await broker.get_clock()
    assert "circuit breaker open" in str(e.value).lower()
    assert calls["n"] == 2  # unchanged

    await broker.close()
