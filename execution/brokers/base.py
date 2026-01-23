from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Optional, Protocol, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt

# Reuse your frozen Step 1 contracts (live ingestion + live execution contracts)
from data_ingestion.contracts import MarketDataEvent, LiveBar
from execution.contracts_live import BrokerClock, OrderRequest, OrderUpdateEvent



# -----------------------------
# Error model (typed + explicit)
# -----------------------------

# in execution/brokers/base.py

@dataclass(frozen=True)
class CallPolicy:
    timeout_s: float = 5.0
    max_attempts: int = 3
    base_backoff_s: float = 1.0
    max_backoff_s: float = 8.0
    breaker_failures: int = 3
    breaker_cooldown_s: float = 60.0


class BrokerError(Exception):
    """Base class for broker adapter errors."""
    pass

class BrokerAuthError(BrokerError):
    """Authentication/authorization failed (e.g., 401/403)."""
    pass


class BrokerUnavailable(BrokerError):
    """Broker or network unavailable (5xx, DNS errors, connection issues, circuit breaker open, etc.)."""
    pass


class BrokerDisconnected(BrokerError):
    pass


class BrokerTimeout(BrokerError):
    pass


class BrokerRateLimited(BrokerError):
    def __init__(self, retry_after_s: float, message: str = "rate limited") -> None:
        super().__init__(message)
        self.retry_after_s = float(retry_after_s)


class BrokerRejected(BrokerError):
    def __init__(self, reason: str, code: Optional[str] = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.code = code


class BrokerInvalidRequest(BrokerError):
    pass


# -----------------------------
# Call policy + circuit breaker
# -----------------------------

@dataclass(frozen=True)
class BrokerCallPolicy:
    timeout_s: float = 5.0
    max_retries: int = 3
    backoff_s: tuple[float, ...] = (1.0, 2.0, 5.0)
    circuit_breaker_failures: int = 3
    circuit_breaker_cooloff_s: float = 60.0


class CircuitBreaker:
    """
    Deterministic circuit breaker:
      - Opens after N failures
      - Closes after cooloff time has elapsed
    """
    def __init__(self, fail_threshold: int, cooloff_s: float) -> None:
        self.fail_threshold = int(fail_threshold)
        self.cooloff_s = float(cooloff_s)
        self._failures = 0
        self._opened_at: Optional[float] = None

    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        # If cooloff elapsed, auto-close
        if (time.time() - self._opened_at) >= self.cooloff_s:
            self.reset()
            return False
        return True

    def record_success(self) -> None:
        self.reset()

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.fail_threshold:
            self._opened_at = time.time()

    def reset(self) -> None:
        self._failures = 0
        self._opened_at = None


# -----------------------------
# Core adapter interface (async)
# -----------------------------

@runtime_checkable
class BrokerAdapter(Protocol):
    """
    Async broker adapter boundary.

    This is NOT wired into ExecutionManager yet. It's the contract that future adapters
    (Alpaca/IBKR) will implement.

    Minimal methods needed for Step 3 acceptance:
      - stream_market_data()
      - submit_order()
    """
    policy: BrokerCallPolicy

    async def connect(self) -> None:
        ...

    async def close(self) -> None:
        ...

    async def get_clock(self) -> BrokerClock:
        ...

    async def stream_market_data(self, *, symbols: list[str]) -> AsyncIterator[MarketDataEvent]:
        ...

    async def submit_order(self, req: OrderRequest) -> OrderUpdateEvent:
        ...


# -----------------------------
# Mock adapter for tests/MVP
# -----------------------------

class MockBrokerAdapter:
    """
    Mock async adapter:
      - stream_market_data(): yields MarketDataEvent objects from an internal asyncio.Queue
      - submit_order(): returns an ACCEPTED OrderUpdateEvent immediately (and optionally emits FILLED)
    """

    def __init__(self, policy: Optional[BrokerCallPolicy] = None) -> None:
        self.policy = policy or BrokerCallPolicy()
        self._connected = False
        self._md_q: asyncio.Queue[MarketDataEvent] = asyncio.Queue()
        self._orders: Dict[str, OrderUpdateEvent] = {}
        self._breaker = CircuitBreaker(
            fail_threshold=self.policy.circuit_breaker_failures,
            cooloff_s=self.policy.circuit_breaker_cooloff_s,
        )

    async def connect(self) -> None:
        self._connected = True

    async def close(self) -> None:
        self._connected = False

    async def get_clock(self) -> BrokerClock:
        # Minimal mock: always open, UTC
        return BrokerClock(
            ts_utc=datetime.now(timezone.utc),
            is_open=True,
            next_open_utc=None,
            next_close_utc=None,
            raw={"mock": True},
        )

    # ---------- Market data plumbing ----------

    async def push_market_event(self, evt: MarketDataEvent) -> None:
        """
        Test helper: push an event into the stream.
        """
        await self._md_q.put(evt)

    async def stream_market_data(self, *, symbols: list[str]) -> AsyncIterator[MarketDataEvent]:
        """
        Yield events, filtered by symbol list. This is infinite until cancelled or closed.
        """
        if not self._connected:
            raise BrokerDisconnected("mock adapter not connected")

        want = {s.upper() for s in symbols}
        while self._connected:
            evt = await self._md_q.get()
            if evt.symbol.upper() in want:
                yield evt

    # ---------- Order submission ----------

    async def submit_order(self, req: OrderRequest) -> OrderUpdateEvent:
        if not self._connected:
            raise BrokerDisconnected("mock adapter not connected")

        if self._breaker.is_open():
            raise BrokerDisconnected("circuit breaker open")

        # Minimal validation that belongs at adapter boundary (not strategy):
        if req.order_type == "LMT" and req.limit_price is None:
            raise BrokerInvalidRequest("limit_price required for LMT")
        if req.order_type == "STP" and req.stop_price is None:
            raise BrokerInvalidRequest("stop_price required for STP")

        # "Accept" immediately
        upd = OrderUpdateEvent(
            broker_order_id="mock_" + uuid4().hex,
            client_order_id=req.client_order_id,
            symbol=req.symbol.upper(),
            status="ACCEPTED",
            side=req.side,
            filled_qty=0,
            avg_fill_price=None,
            last_fill=None,
            event_ts_utc=datetime.now(timezone.utc),
            raw={"mock": True, "order_type": req.order_type, "tif": req.tif},
        )
        self._orders[req.client_order_id] = upd
        self._breaker.record_success()
        return upd
