# -----------------------------------------------------------------------------
# 2) scanner/sector_feed.py  – async price cache for ETF / index futures
# -----------------------------------------------------------------------------

"""Live sector / index price cache shared by SectorMomentumDetector.

For production you would connect to a broker / data‑vendor WebSocket.  Here we
ship a minimal stub (random walk) that keeps last price per *symbol* updated
once per second.  Detectors can query `SectorPriceCache.get()` with zero
latency.
"""
from collections import defaultdict
import asyncio
import random
from typing import Dict, Optional

class SectorPriceCache:
    """Thread‑safe in‑memory last‑price store."""

    def __init__(self) -> None:
        self._price: Dict[str, float] = defaultdict(lambda: float("nan"))
        self._lock = asyncio.Lock()

    async def update(self, sym: str, px: float) -> None:  # noqa: D401
        async with self._lock:
            self._price[sym] = px

    async def get(self, sym: str) -> Optional[float]:
        async with self._lock:
            return self._price.get(sym)

CACHE = SectorPriceCache()

async def stub_sector_loop(symbols: list[str], *, refresh: float = 1.0) -> None:
    """Dummy random‑walk price generator for offline dev / CI."""
    rng = random.Random(0)
    # Initialise with 100.0
    for s in symbols:
        await CACHE.update(s, 100.0)
    while True:
        for s in symbols:
            last = await CACHE.get(s) or 100.0
            await CACHE.update(s, last * (1.0 + rng.uniform(-0.0005, 0.0005)))
        await asyncio.sleep(refresh)
