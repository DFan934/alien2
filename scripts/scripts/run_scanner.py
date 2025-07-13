# ---------------------------------------------------------------------------
# FILE: scripts/run_scanner.py
# ---------------------------------------------------------------------------
"""Command‑line entry‑point to run the scanner in *live* or *backtest* mode.

Example invocations:
    # Live (bars fed by another process)
    python -m scripts.run_scanner --mode live --redis redis://localhost:6379/0 \
        --parquet-root parquet/scanner_events

    # Back‑test from already engineered minute Parquet
    python -m scripts.run_scanner --mode backtest --parquet parquet/minute/AAPL_2023.parquet \
        --symbol AAPL
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import List

import pandas as pd

from scanner.detectors import CompositeDetector, GapDetector, HighRVOLDetector#, BullishMomentumDetector
from scanner.recorder import DataGroupBuilder
from scanner.live_loop import ScannerLoop
from scanner.backtest_loop import BacktestScannerLoop


def build_detectors(args) -> CompositeDetector:  # noqa: D401
    return CompositeDetector([
        GapDetector(pct=args.gap_pct, direction="both"),
        HighRVOLDetector(thresh=args.rvol),
        #BullishMomentumDetector(window=args.mom_window),
    ])


def main(argv: List[str] | None = None) -> None:  # noqa: D401
    p = argparse.ArgumentParser(description="Run the stock setup scanner")
    sub = p.add_subparsers(dest="mode", required=True)

    live = sub.add_parser("live", help="Run scanner in live async mode")
    live.add_argument("--redis", required=False, default=None, help="Redis connection URL for event fan‑out")
    live.add_argument("--parquet-root", required=True, help="Output parquet root for snapshots")
    live.add_argument("--gap-pct", type=float, default=0.02)
    live.add_argument("--rvol", type=float, default=2.0)
    live.add_argument("--mom-window", type=int, default=3)
    live.add_argument("--refresh", type=float, default=5.0, help="Refresh interval seconds")

    bt = sub.add_parser("backtest", help="Run scanner over historical parquet file")
    bt.add_argument("--parquet", required=True, help="Parquet file with historical minute bars")
    bt.add_argument("--parquet-root", required=True, help="Output root for scanner snapshots")
    bt.add_argument("--gap-pct", type=float, default=0.02)
    bt.add_argument("--rvol", type=float, default=2.0)
    bt.add_argument("--mom-window", type=int, default=3)

    args = p.parse_args(argv)

    detectors = build_detectors(args)
    builder = DataGroupBuilder(args.parquet_root, redis_url=getattr(args, "redis", None))

    if args.mode == "live":
        bar_q: asyncio.Queue = asyncio.Queue(maxsize=1024)
        loop = ScannerLoop(detectors, builder, bar_q, refresh_sec=args.refresh)
        loop.start()
        print("[Scanner] Live mode started – waiting for bars …")
        asyncio.get_event_loop().run_forever()
    else:  # backtest
        df = pd.read_parquet(args.parquet)
        bts = BacktestScannerLoop(detectors, builder, df)
        cnt = sum(1 for _ in bts)  # iterate and persist
        print(f"[Scanner][Backtest] Recorded {cnt} scanner events → {args.parquet_root}")
        builder.flush()


if __name__ == "__main__":
    main()
