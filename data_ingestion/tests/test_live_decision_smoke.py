# tests/test_live_decisions_smoke.py
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from data_ingestion.live.runtime import LiveRuntime, LiveRuntimeConfig


REPO_ROOT = Path(__file__).resolve().parents[2]  # .../TheFinalProject5


@pytest.mark.asyncio
async def test_live_runtime_writes_outputs(tmp_path: Path):
    """
    Fast smoke test: runs LiveRuntime briefly with a dummy feed and checks
    that snapshots/decisions/latency datasets have at least one part file.
    """
    # NOTE: You must point these at real folders in your environment when running locally.
    # In CI, you can skip this test or provide tiny test fixtures for ev_root/fe_root.
    #ev_root = Path("prediction_engine/artifacts")  # change to your real EV folder

    #ev_root = Path("models/ev_live")

    #fe_root = Path("prediction_engine/artifacts")  # change to your real FE folder containing _fe_meta/pipeline.pkl

    #fe_root = Path("models/fe_live")

    ev_root = REPO_ROOT / "models" / "ev_live"
    fe_root = REPO_ROOT / "models" / "fe_live"

    print("REPO_ROOT =", REPO_ROOT)
    print("ev_root   =", ev_root, "exists?", ev_root.exists())
    print("fe_root   =", fe_root, "exists?", fe_root.exists())

    if not (ev_root.exists() and fe_root.exists()):
        pytest.skip("ev_root/fe_root not available in test environment")

    bar_queue: "asyncio.Queue[list[dict]]" = asyncio.Queue()

    async def dummy_feed():
        import random, time
        price = 100.0
        while True:
            price *= 1 + random.uniform(-0.001, 0.001)
            bar = {
                "symbol": "AAPL",
                "timestamp": time.time(),
                "open": price,
                "high": price * 1.001,
                "low": price * 0.999,
                "close": price,
                "volume": 1000,
            }
            await bar_queue.put([bar])
            await asyncio.sleep(0.2)

    rt = LiveRuntime(
        cfg=LiveRuntimeConfig(out_dir=tmp_path, refresh_s=1.0, ev_root=ev_root, fe_root=fe_root),
        bar_queue=bar_queue,
    )

    feed_task = asyncio.create_task(dummy_feed())

    # run ~3 seconds then stop
    try:
        await rt.run(stop_after_s=3.0)
    except Exception:
        # LiveRuntime uses TaskGroup; stopper raises KeyboardInterrupt inside group.
        pass
    finally:
        feed_task.cancel()

    # Verify datasets exist and have at least one part
    assert any((tmp_path / "snapshots.parquet").glob("part-*.parquet"))
    assert any((tmp_path / "decisions.parquet").glob("part-*.parquet"))
    assert any((tmp_path / "latency_report.parquet").glob("part-*.parquet"))
