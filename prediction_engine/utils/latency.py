# ---------------------------------------------------------------------------
# prediction_engine/utils/latency.py
# ---------------------------------------------------------------------------
"""Latency & Prometheus helpers
================================
Provides a simple `@timeit(stage)` decorator that records execution latency for
both sync and async functions and exposes them via `prometheus_client`.
"""
from __future__ import annotations

import inspect
import time
from functools import wraps

from prometheus_client import Summary

# One Summary metric with a *stage* label so we can slice by pipeline section.
REQUEST_LATENCY = Summary(
    "prediction_engine_latency_seconds",
    "Latency of critical pipeline sections",
    ["stage"],
)

def timeit(stage: str):
    """Decorator to record execution latency under the given *stage* label.

    Works transparently for both synchronous and asynchronous callables.
    """
    def decorator(fn):
        meter = REQUEST_LATENCY.labels(stage=stage)

        if inspect.iscoroutinefunction(fn):

            @wraps(fn)
            async def _async_wrapped(*args, **kwargs):
                tic = time.perf_counter()
                try:
                    return await fn(*args, **kwargs)
                finally:
                    meter.observe(time.perf_counter() - tic)

            return _async_wrapped

        @wraps(fn)
        def _sync_wrapped(*args, **kwargs):
            tic = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                meter.observe(time.perf_counter() - tic)

        return _sync_wrapped

    return decorator
