# ──────────────────────────────────────────────────────────────────────────
# feature_engineering/utils.py
# ──────────────────────────────────────────────────────────────────────────
"""Logger + timing decorator shared across feature‑engineering sub‑package."""
from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Callable, TypeVar

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Logger – configured only if root logger has no handlers (library‑friendly)
# ---------------------------------------------------------------------------
logger = logging.getLogger("feature_engineering")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Simple timing decorator (mirrors one in data_ingestion.utils)
# ---------------------------------------------------------------------------

def timeit(name: str | None = None, *, level: int = logging.INFO) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Log execution time of *fn* at *level*."""

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        disp = name or fn.__qualname__

        @wraps(fn)
        def wrapper(*args, **kwargs):  # type: ignore[override]
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            logger.log(level, "%s finished in %.2fs", disp, time.perf_counter() - t0)
            return result

        return wrapper



    return decorator



# (End utils.py)