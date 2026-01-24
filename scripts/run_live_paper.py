# scripts/run_live_paper.py
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Optional

from data_ingestion.historical.live_writer import now_run_id
from data_ingestion.live.runtime import LiveRuntime, LiveRuntimeConfig


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser("Run live decisions-only pipeline (no orders)")

    # Where to write outputs (NOT artifacts/)
    p.add_argument("--out", default=str(Path("runs") / "live" / now_run_id()))

    # Where to load EV model artifacts from (can be any dir; NOT required to be artifacts/)
    p.add_argument("--ev-root", required=True, help="Folder used by EVEngine.from_artifacts(...)")

    # Where to load frozen FE pipeline from (must contain _fe_meta/pipeline.pkl)
    p.add_argument("--fe-root", required=True, help="Folder containing _fe_meta/pipeline.pkl for CoreFeaturePipeline")

    # Scanner refresh
    p.add_argument("--refresh", type=float, default=5.0)

    # Optional short smoke run for testing
    p.add_argument("--stop-after-s", type=float, default=None)

    # If you want to run without Step5 temporarily, enable dummy feed
    p.add_argument("--dummy-feed", action="store_true", help="Use synthetic bars instead of Step 5 market feed")
    return p.parse_args()


async def _dummy_feed(bar_queue: "asyncio.Queue[list[dict]]") -> None:
    import random
    import time
    price = 150.0
    sym = "AAPL"
    while True:
        price *= 1 + random.uniform(-0.0008, 0.0008)
        ts = time.time()
        bar = {
            "symbol": sym,
            "timestamp": ts,   # ScannerLoop may accept either; your step5 finalizer uses timestamp objects
            "open": price,
            "high": price * 1.0008,
            "low": price * 0.9992,
            "close": price,
            "volume": random.randint(1000, 9000),
        }
        await bar_queue.put([bar])  # ScannerLoop expects list of bars
        await asyncio.sleep(1.0)


async def main() -> None:
    args = _parse()
    out_dir = Path(args.out)

    # Bar queue that ScannerLoop reads from (Step 5 will feed this)
    bar_queue: "asyncio.Queue[list[dict]]" = asyncio.Queue()

    if args.dummy_feed:
        asyncio.create_task(_dummy_feed(bar_queue))
    else:
        # Step 5 should be running and injecting into this bar_queue.
        # If you implemented Step 5 as a publisher that takes a bar_queue,
        # you should start it here (or in a wrapper script).
        pass

    rt = LiveRuntime(
        cfg=LiveRuntimeConfig(
            out_dir=out_dir,
            refresh_s=float(args.refresh),
            fe_root=Path(args.fe_root),
            ev_root=Path(args.ev_root),
        ),
        bar_queue=bar_queue,
    )

    await rt.run(stop_after_s=args.stop_after_s)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[run_live_paper] shutdown requested – bye")
