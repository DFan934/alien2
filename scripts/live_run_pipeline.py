# -------------------------------------------------------------
# File: scripts2/run_live_pipeline.py  (UPDATED)
# -------------------------------------------------------------
"""End‑to‑end live pipeline launcher.

With sensible defaults bundled in, you can simply run:

    python scripts2/run_live_pipeline.py

and the script will spin up:
  • ScannerLoop (gap + RVOL + momentum detectors)
  • RedisScannerEventConsumer  → ExecutionManager
  • ExecutionManager   (writes signals to logs/signals.log)

Optional CLI flags *still* exist to override redis URL, etc., but none are
required.
"""
from __future__ import annotations

import asyncio
import argparse
import logging
from pathlib import Path

from scanner.live_loop import ScannerLoop  # type: ignore
from scanner.detectors import PREMARKET_RULES  # type: ignore
from scanner.recorder import DataGroupBuilder  # type: ignore
from prediction_engine.stream.redis_consumer import RedisScannerEventConsumer
from prediction_engine.ev_engine import EVEngine
from prediction_engine.execution.risk_manager import RiskManager  # type: ignore
from prediction_engine.execution.manager import ExecutionManager
from prediction_engine.utils.latency import timeit  # simple exp‑avg helper

# ------------------------- defaults --------------------------
DEFAULT_REDIS = "redis://localhost:6379"
DEFAULT_STREAM = "scanner:events"
DEFAULT_GROUP = "live-pipeline"
DEFAULT_CONSUMER = "exec-1"
DEFAULT_REFRESH = 5.0
DEFAULT_ARTIFACTS = "prediction_engine/artifacts"

# ------------------------- CLI (all optional) ----------------

def _parse() -> argparse.Namespace:  # noqa: D401
    p = argparse.ArgumentParser("Run full live pipeline – all args optional")
    p.add_argument("--redis", default=DEFAULT_REDIS)
    p.add_argument("--stream", default=DEFAULT_STREAM)
    p.add_argument("--group", default=DEFAULT_GROUP)
    p.add_argument("--consumer", default=DEFAULT_CONSUMER)
    p.add_argument("--refresh", type=float, default=DEFAULT_REFRESH)
    p.add_argument("--artefacts", default=DEFAULT_ARTIFACTS)
    return p.parse_args()


# ------------------------- main orchestration ---------------

async def main() -> None:  # noqa: D401
    args = _parse()

    # --- instantiate EVEngine -----------------------------------
    ev = EVEngine.from_artifacts(args.artefacts)

    # --- latency monitor stub -----------------------------------
    latency = timeit()

    # --- risk manager -------------------------------------------
    risk = RiskManager(account_equity=100_000)
    exec_mgr = ExecutionManager(ev, risk, latency, config={}, log_path=Path("logs/signals.log"))
    await exec_mgr.start()

    # --- bar feed / scanner -------------------------------------
    # In real deployment, bar_queue is wired to your WebSocket feed.
    # Here we provide a stub async generator that yields synthetic bars.
    bar_queue: "asyncio.Queue[dict]" = asyncio.Queue()

    async def _dummy_feed() -> None:  # stub – replace with real WS ingest
        import random, time
        sym = "AAPL"
        price = 150.0
        while True:
            price *= 1 + random.uniform(-0.001, 0.001)
            bar = {
                "symbol": sym,
                "price": price,
                "open": price,
                "high": price * 1.001,
                "low": price * 0.999,
                "volume": random.randint(1000, 5000),
                "ts": time.time(),
            }
            await bar_queue.put([bar])  # ScannerLoop expects list of bars
            await asyncio.sleep(1.0)

    asyncio.create_task(_dummy_feed())

    builder = DataGroupBuilder(parquet_root=Path("scanner_events"))
    scanner = ScannerLoop(detectors=PREMARKET_RULES, builder=builder, bar_queue=bar_queue, refresh=args.refresh)
    asyncio.create_task(scanner.run())

    # --- redis consumer -----------------------------------------
    consumer = RedisScannerEventConsumer(
        redis_url=args.redis,
        stream=args.stream,
        group=args.group,
        consumer=args.consumer,
        exec_mgr=exec_mgr,
    )
    await consumer.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[Pipeline] shutdown requested – bye")
