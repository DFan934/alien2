# ---------------------------------------------------------------------------
# FILE: scanner/live_loop.py
# ---------------------------------------------------------------------------
"""Asynchronous runner for live feeds – expects bars injected into
an asyncio.Queue from a WebSocket client.  Each *item* is a list of dicts
with keys matching our standard OHLCV schema.
"""
from __future__ import annotations

import asyncio
from typing import Iterable, Mapping

import pandas as pd

from .detectors import CompositeDetector
from .recorder import DataGroupBuilder

__all__ = ["ScannerLoop"]


class ScannerLoop:
    def __init__(
        self,
        detectors: CompositeDetector,
        builder: DataGroupBuilder,
        bar_queue: asyncio.Queue,  # list[dict] batches
        refresh_sec: float = 5.0,
    ) -> None:
        self.detectors = detectors
        self.builder = builder
        self.bar_q = bar_queue
        self.refresh_sec = refresh_sec
        self._task: asyncio.Task | None = None

    # ----------------------------------------------
    # Public lifecycle
    # ----------------------------------------------
    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="scanner_loop")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    # ----------------------------------------------
    # Internal
    # ----------------------------------------------
    async def _run(self) -> None:
        while True:
            batch: Iterable[Mapping] = await self.bar_q.get()
            df = pd.DataFrame(batch).set_index("timestamp")
            # CompositeDetector is *async* ‑‑ need to await
            mask = await self.detectors(df)
            if mask.any():
                snaps = df[mask]
                for ts, row in snaps.iterrows():
                    await self.builder.log(ts, row["symbol"], row)
            await asyncio.sleep(self.refresh_sec)