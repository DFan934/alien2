# -----------------------------------------------------------------------------
# 7) scanner/tests/test_sector_feed.py (stub)
# -----------------------------------------------------------------------------

import asyncio, pytest
from scanner.sector_feed import CACHE, stub_sector_loop

@pytest.mark.asyncio
async def test_stub_sector_cache():
    task = asyncio.create_task(stub_sector_loop(["XLE"], refresh=0.01))
    await asyncio.sleep(0.05)
    price = await CACHE.get("XLE")
    task.cancel()
    assert price is not None and price == price  # not NaN