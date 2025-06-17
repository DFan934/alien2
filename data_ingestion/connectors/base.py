# ========================
# file: data_ingestion/connectors/base.py
# ========================
"""Abstract interface every connector must implement."""

from __future__ import annotations

import abc
import asyncio
import logging
from datetime import datetime
from typing import List, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["BaseConnector"]


class ProviderAuthError(RuntimeError):
    """Raised when a provider rejects our credentials (HTTP 401/403)."""


class BaseConnector(abc.ABC):
    """A provider‑specific, *stateless* I/O adapter."""

    NAME: str  # concrete provider name (lowercase)

    #: Set of time‑frame strings the provider can return exactly ("1Min", "daily", ...)
    TIMEFRAMES: set[str]

    #: Optional multiplier that converts provider‑native *lot* sizes → shares
    SIZE_MULTIPLIER: float = 1.0

    def __init__(self, **config):
        self.config = config

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------

    @classmethod
    def supports_timeframe(cls, tf: str) -> bool:
        return tf in cls.TIMEFRAMES

    # ------------------------------------------------------------------
    # Mandatory API
    # ------------------------------------------------------------------

    @abc.abstractmethod
    async def fetch_data(
        self,
        symbols: Sequence[str],
        timeframe: str,
        start_date: str | datetime,
        end_date: str | datetime,
    ) -> pd.DataFrame:
        """Return raw *provider‑native* rows for *symbols*.

        The DataFrame **must not** perform renames or timezone alignment – that
        is handled downstream by pipelines.
        Columns *MUST* include at minimum:
        ``timestamp, open, high, low, close, volume`` (case preserved).
        """

    # ------------------------------------------------------------------
    # Helpers for synchronous backends
    # ------------------------------------------------------------------

    async def _run_sync(self, fn, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, fn, *args, **kwargs)