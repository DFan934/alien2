# ---------------------------------------------------------------------------
# FILE: scanner/recorder.py  (REPLACED)
# ---------------------------------------------------------------------------
"""Persist snapshot rows that qualified the detector mask.
   • Parquet hive‑partitioned for back‑tests
   • (optional) Redis Stream for live fan‑out to EVEngine & dashboards

This version adds a **log_sync** helper so tests and the historical replay
loop never require `asyncio`.
"""
from __future__ import annotations

import os
from datetime import timezone
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    import redis.asyncio as redis  # type: ignore
except ImportError:  # Redis optional – graceful‑degrade
    redis = None  # type: ignore

__all__ = ["DataGroupBuilder"]


class DataGroupBuilder:
    """Append each *snapshot* row to Parquet and optionally stream to Redis.

    Parameters
    ----------
    parquet_root
        Base directory for hive‑partitioned snapshots.
    redis_url
        Optional Redis URL for live streaming.  Not used in back‑tests.
    """

    def __init__(self, parquet_root: str | os.PathLike, redis_url: str | None = None, buffer_size: int = 5_000):
        self.parquet_root = Path(parquet_root)
        self.parquet_root.mkdir(parents=True, exist_ok=True)
        self._redis = None
        if redis_url:
            if redis is None:
                raise RuntimeError("redis-py not installed but redis_url supplied")
            self._redis = redis.from_url(redis_url, decode_responses=False)
        self._buf = []                # buffer for snapshots
        self._buffer_size = buffer_size


    # --------------------------------------------------
    # 🔑 Public API  –  async (live) *and* sync (back‑test)
    # --------------------------------------------------
    async def log(self, ts: pd.Timestamp, symbol: str, snapshot: pd.Series) -> None:
        """Persist one snapshot row – *async‑friendly* for live mode."""
        self._to_parquet(symbol, snapshot)
        if self._redis is not None:
            await self._to_redis(ts, symbol, snapshot)

    def log_sync(self, ts: pd.Timestamp, symbol: str, snapshot: pd.Series) -> None:
        """Buffer snapshots in memory and flush as a batch."""
        snap = snapshot.copy()
        snap["symbol"] = symbol
        snap["timestamp"] = ts
        self._buf.append(snap)
        if len(self._buf) >= self._buffer_size:
            self._flush()

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------
    def _to_parquet(self, symbol: str, snap: pd.Series) -> None:
        dt = snap.name.tz_convert(timezone.utc).to_pydatetime()
        part = (
            self.parquet_root
            / f"symbol={symbol}"
            / f"year={dt.year}"
            / f"month={dt.month:02}"
        )
        part.mkdir(parents=True, exist_ok=True)
        file = part / f"{dt:%Y%m%dT%H%M%S}.parquet"
        pd.DataFrame([snap]).to_parquet(file, index=False)

    async def _to_redis(self, ts: pd.Timestamp, symbol: str, snap: pd.Series) -> None:  # noqa: D401
        payload = snap.to_json(date_format="iso", date_unit="s").encode()
        stream = f"scanner:events:{symbol}"
        assert self._redis is not None  # mypy guard
        await self._redis.xadd(stream, {b"data": payload})

    def _flush(self):
        """Write all buffered snapshots to disk as one batch."""
        if not self._buf:
            return
        df = pd.DataFrame(self._buf)
        # For demo, write a single Parquet file (real code could partition by symbol/month)
        out_file = self.parquet_root / "scanner_events.parquet"
        df.to_parquet(out_file, append=True)
        self._buf.clear()

    def flush(self):
        """Explicit flush for end-of-backtest/teardown."""
        self._flush()