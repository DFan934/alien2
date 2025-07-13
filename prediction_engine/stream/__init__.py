# ---------------------------------------------------------------------------
# prediction_engine/stream/__init__.py
# ---------------------------------------------------------------------------

"""Streaming helpers (Redis, Kafka, etc.).

Right now we ship only a Redis consumer that converts Scanner events into the
`bar` dict format expected by :py:meth:`ExecutionManager.on_bar`.
"""

from importlib import metadata as _md
__all__ = ("RedisScannerEventConsumer",)
__version__ = _md.version("prediction_engine") if _md else "0.0.0"

from .redis_consumer import RedisScannerEventConsumer  # noqa: E402  (forward import)