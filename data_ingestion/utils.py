############################
# data_ingestion/utils.py
############################
"""Shared helpers: logger, timer and YAML config loader.
This rev adds a *robust* `load_config` that searches multiple default
locations so you don’t get a FileNotFoundError when running from any cwd.
"""
from __future__ import annotations

import logging
import os
import time
from functools import wraps
from pathlib import Path
from typing import Callable, Optional, Union

import yaml

# ----------------------------------------------------------------------------
# Logging setup (root‑level logger can be overridden by app if desired)
# ----------------------------------------------------------------------------
logger = logging.getLogger("data_ingestion")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

# ----------------------------------------------------------------------------
# Config loader – searches explicit arg → env var → CWD → package root.
# ----------------------------------------------------------------------------
_DEFAULT_CONFIG_NAME = "ingest_config.yaml"
_ENV_VAR = "INGEST_CONFIG_PATH"


def _possible_paths(explicit: Optional[Union[str, Path]]) -> list[Path]:
    paths: list[Path] = []
    if explicit:
        paths.append(Path(explicit).expanduser())
    env_path = os.getenv(_ENV_VAR)
    if env_path:
        paths.append(Path(env_path).expanduser())
    paths.extend([
        Path.cwd() / _DEFAULT_CONFIG_NAME,
        Path(__file__).resolve().parent.parent / _DEFAULT_CONFIG_NAME,
    ])
    # De‑duplicate while preserving order
    seen = set()
    ordered: list[Path] = []
    for p in paths:
        if p not in seen:
            ordered.append(p)
            seen.add(p)
    return ordered


def load_config(path: Optional[Union[str, Path]] = None) -> dict:
    """Locate and load *ingest_config.yaml*.

    Raises
    ------
    FileNotFoundError
        If no YAML file is found in the search sequence.
    """
    for candidate in _possible_paths(path):
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            logger.info("Loaded ingest config from %s", candidate)
            return cfg or {}
    msg = "Ingest config not found. Looked for: " + ", ".join(
        map(str, _possible_paths(path))
    )
    raise FileNotFoundError(msg)

# ----------------------------------------------------------------------------
# Flexible `timeit` decorator (can be called with or without label)
# ----------------------------------------------------------------------------

def timeit(_fn: Optional[Callable] = None, *, label: Optional[str] = None) -> Callable:
    """Decorator to log wall‑time.  Usage::

        @timeit                        # label defaults to function.__name__
        def foo(): ...

        @timeit(label="ingestion‑run") # explicit label
        def bar(): ...
    """

    def decorator(fn: Callable) -> Callable:  # type: ignore[arg‑type]
        _lbl = label or fn.__name__

        @wraps(fn)
        def wrapper(*args, **kwargs):  # type: ignore[override]
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                logger.info("%s took %.3f s", _lbl, duration)

        return wrapper

    if _fn is not None and callable(_fn):
        # Used as bare decorator @timeit
        return decorator(_fn)
    # Used as factory @timeit(label="...") or @timeit()
    return decorator
