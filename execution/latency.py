# -----------------------------------------------------------------------------
# File: execution/latency.py
# -----------------------------------------------------------------------------
# execution/latency.py
"""Utility decorator + rolling stats for call latency."""
from __future__ import annotations

import time
import inspect
from collections import deque
from functools import wraps
from typing import Callable, Deque, Dict
import numpy as np


class LatencyMonitor:
    """Maintains rolling latency (ms) statistics per label."""

    def __init__(self, maxlen: int = 1_000):
        self._buf: Dict[str, Deque[float]] = {}
        self._maxlen = maxlen

    def record(self, label: str, ms: float) -> None:
        q = self._buf.setdefault(label, deque(maxlen=self._maxlen))
        q.append(ms)

    def mean(self, label: str) -> float:
        q = self._buf.get(label, [])
        return sum(q) / len(q) if q else 0.0

    def p95(self, label: str) -> float:
        q = self._buf.get(label, [])
        return float(np.percentile(q, 95)) if q else 0.0

    # --- small helpers for tests/metrics ---
    def count(self, label: str) -> int:
        q = self._buf.get(label, [])
        return len(q)

    def reset(self, label: str | None = None) -> None:
        if label is None:
            self._buf.clear()
        else:
            self._buf.pop(label, None)


latency_monitor = LatencyMonitor()


def timeit(label: str) -> Callable:
    """
    Decorator to record latency (ms) of the wrapped call.
    Works for both sync and async functions.
    """
    def decorator(func: Callable):
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def _async_wrapped(*args, **kwargs):
                t0 = time.perf_counter()
                try:
                    return await func(*args, **kwargs)
                finally:
                    dt_ms = (time.perf_counter() - t0) * 1_000
                    latency_monitor.record(label, dt_ms)
            return _async_wrapped

        @wraps(func)
        def _sync_wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                dt_ms = (time.perf_counter() - t0) * 1_000
                latency_monitor.record(label, dt_ms)
        return _sync_wrapped

    return decorator
