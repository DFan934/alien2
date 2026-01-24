# data_ingestion/tests/test_live_publisher_step5.py
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List

import pandas as pd
import pytest

from data_ingestion.live.publisher import MarketDataPublisher


class FakeWSClient:
    def __init__(self, events: List[Dict[str, Any]]) -> None:
        self._events = events

    async def stream_raw(self, *, symbols: list[str]) -> AsyncIterator[Dict[str, Any]]:
        for ev in self._events:
            yield ev

    async def close(self) -> None:
        return


@pytest.mark.asyncio
async def test_step5_publisher_writes_parquet_and_injects_queue(tmp_path: Path):
    # Two minutes: seeing minute 09:31 finalizes 09:30 (BarFinalizer watermark logic)
    t0 = datetime(2026, 1, 22, 14, 30, 30, tzinfo=timezone.utc).isoformat()
    t1 = datetime(2026, 1, 22, 14, 31,  5, tzinfo=timezone.utc).isoformat()

    fake_events = [
        {"T": "b", "S": "AAPL", "t": t0, "o": 10, "h": 11, "l": 9, "c": 10.5, "v": 100},
        {"T": "b", "S": "AAPL", "t": t1, "o": 10.5, "h": 12, "l": 10, "c": 11.0, "v": 120},
    ]

    q: asyncio.Queue = asyncio.Queue()
    cfg = {
        "live": {
            "emit_gaps": True,
            "flush_every_n_raw": 1,
            "flush_every_n_final": 1,
        },
        "broker": {"alpaca": {"key_id": "x", "secret_key": "y"}},  # not used by FakeWSClient
    }

    pub = MarketDataPublisher(
        cfg=cfg,
        out_dir=tmp_path,
        bar_queue=q,
        symbols=["AAPL"],
        ws_client=FakeWSClient(fake_events),  # inject mock
    )

    await pub.run(max_raw_messages=len(fake_events))

    # queue should have received at least one finalized batch
    assert q.qsize() >= 1
    batch = await q.get()
    assert isinstance(batch, list)
    assert len(batch) >= 1
    assert "timestamp" in batch[0]
    assert "symbol" in batch[0]

    # artifacts exist and have rows
    raw_fp = tmp_path / "bars_raw.parquet"
    fin_fp = tmp_path / "bars_final.parquet"
    assert raw_fp.exists()
    assert fin_fp.exists()

    raw_df = pd.read_parquet(raw_fp)
    fin_df = pd.read_parquet(fin_fp)
    assert len(raw_df) >= 1
    assert len(fin_df) >= 1
    for col in ["timestamp", "symbol", "open", "high", "low", "close", "volume", "bar_present", "is_gap"]:
        assert col in fin_df.columns
