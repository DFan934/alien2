# -----------------------------------------------------------------------------
# File: execution/broker_api.py
# -----------------------------------------------------------------------------
"""Abstract Broker API and a synchronous MockBroker for tests."""
from __future__ import annotations

import abc
from datetime import datetime
from typing import Callable, Dict, List

from core.contracts import FillEvent, OrderEvent


class BrokerAPI(abc.ABC):
    """Minimal broker interface – async not used in M0 to keep tests simple."""

    on_fill: Callable[[FillEvent], None]

    @abc.abstractmethod
    def submit_order(self, order: OrderEvent) -> None:  # noqa: D401
        """Submit order and *eventually* call self.on_fill(FillEvent)."""
        raise NotImplementedError


class MockBroker(BrokerAPI):
    """Immediate‑fill, zero‑latency mock for unit tests."""

    def __init__(self, price_lookup: Callable[[str], float]):
        self._price_lookup = price_lookup
        self.on_fill: Callable[[FillEvent], None] = lambda fill: None
        self._order_id = 0

    def submit_order(self, order: OrderEvent) -> None:
        self._order_id += 1
        fill_price = self._price_lookup(order.symbol)
        fill = FillEvent(
            order_id=str(self._order_id),
            signal_id=order.signal_id,
            symbol=order.symbol,
            side=order.side,
            fill_px=fill_price,
            qty=order.qty,
            timestamp=datetime.utcnow(),
        )
        
        # immediate callback – sync for M0
        self.on_fill(fill)
