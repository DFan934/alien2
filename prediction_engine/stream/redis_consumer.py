# -------------------------------------------------------------
# File: prediction_engine/stream/redis_consumer.py
# -------------------------------------------------------------
"""Async Redis Stream → ExecutionManager bridge."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

import aioredis
import pandas as pd


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RedisScannerEventConsumer:
    redis_url: str = "redis://localhost:6379"
    stream: str = "scanner:events"
    group: str = "live-pipeline"
    consumer: str = "exec-1"
    exec_mgr: Any | None = None  # expects ExecutionManager interface
    batch: int = 50
    block_ms: int = 2000

    async def _connect(self) -> aioredis.Redis:  # type: ignore
        r = await aioredis.from_url(self.redis_url, decode_responses=True)
        # Auto‑create group if missing
        try:
            await r.xgroup_create(self.stream, self.group, id="0-0", mkstream=True)
        except aioredis.ResponseError as e:  # group exists
            if "BUSYGROUP" not in str(e):
                raise
        return r

    async def run(self) -> None:
        redis = await self._connect()
        last_id = ">"  # consumer‑group style; only new msgs
        while True:
            msgs = await redis.xreadgroup(self.group, self.consumer, {self.stream: last_id}, count=self.batch, block=self.block_ms)
            if not msgs:
                continue
            # msgs: List[(stream, [(id, {field: value})])]
            for _stream, entries in msgs:
                for msg_id, data in entries:
                    try:
                        await self._handle(data)
                    finally:
                        await redis.xack(self.stream, self.group, msg_id)

    # ------------------------------------------------------
    async def _handle(self, data: dict[str, str]) -> None:
        """Transform scanner event → ExecutionManager.on_bar() call."""
        if self.exec_mgr is None:
            return
        # scanner writes JSON string under field "snap"
        snap = json.loads(data["snap"])
        await self.exec_mgr.on_bar(snap)
