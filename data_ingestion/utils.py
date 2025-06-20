############################
# data_ingestion/utils.py
############################
"""Shared helpers (logging & timing)."""
import logging, time, functools, pathlib

LOGGER_NAME = "data_ingestion"
logger = logging.getLogger(LOGGER_NAME)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
    logger.setLevel(logging.INFO)
    logger.addHandler(h)


def timeit(label: str):
    """Decorator to time a function call and log duration."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            result = fn(*args, **kwargs)
            logger.info(f"{label} finished in {time.time()-t0:.2f}s")
            return result
        return wrapper
    return decorator