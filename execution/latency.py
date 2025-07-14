# -----------------------------------------------------------------------------
# File: execution/latency.py
# -----------------------------------------------------------------------------
"""Utility decorator + rolling stats for call latency."""
from __future__ import annotations

import time
from collections import deque
from functools import wraps
from typing import Callable, Deque, Dict, Tuple
import numpy as np      # <- put below existing imports


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
        """Return 95-th percentile latency for *label* over current window."""
        q = self._buf.get(label, [])
        return float(np.percentile(q, 95)) if q else 0.0


latency_monitor = LatencyMonitor()


def timeit(label: str) -> Callable:
    """Decorator to record latency of the wrapped call."""

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            res = func(*args, **kwargs)
            dt_ms = (time.perf_counter() - t0) * 1_000
            latency_monitor.record(label, dt_ms)
            return res

        return wrapper

    return decorator