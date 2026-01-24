# data_ingestion/live/runtime.py
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from data_ingestion.historical.live_writer import LiveParquetWriter, ensure_dir, write_run_manifest
from feature_engineering.pipelines.core import CoreFeaturePipeline
from prediction_engine.ev_engine import EVEngine
from prediction_engine.scoring.batch import vectorize_minute_batch

from scanner.detectors import build_detectors
from scanner.live_loop import ScannerLoop


class InProcSnapshotBuilder:
    """
    Minimal async builder compatible with scanner/live_loop.py:
      await builder.log(ts, symbol, row)

    Writes snapshots continuously and pushes them into snap_queue for decisions-only mode.
    """
    def __init__(
        self,
        snapshots_writer: LiveParquetWriter,
        snap_queue: "asyncio.Queue[Dict[str, Any]]",
        max_queue: int = 20000,
    ) -> None:
        self.snapshots_writer = snapshots_writer
        self.snap_queue = snap_queue
        self.max_queue = int(max_queue)

    async def log(self, ts, symbol: str, row) -> None:
        r = dict(row)
        r["timestamp"] = ts
        r["symbol"] = symbol

        self.snapshots_writer.append(pd.DataFrame([r]))

        # bounded queue (drop-oldest to keep memory stable)
        try:
            if self.snap_queue.qsize() >= self.max_queue:
                _ = self.snap_queue.get_nowait()
            self.snap_queue.put_nowait(r)
        except Exception:
            pass


@dataclass
class LiveRuntimeConfig:
    out_dir: Path
    refresh_s: float = 5.0
    max_snap_queue: int = 20000

    # IMPORTANT: these can point anywhere; they do NOT need to be "artifacts/"
    fe_root: Path | None = None
    ev_root: Path | None = None

    # Optional detector config path (YAML); if None, uses detector defaults
    detectors_yaml: str | None = None
    dev_loose_detectors: bool = False  # useful during bring-up


class LiveRuntime:
    """
    Step 6: Decisions-only mode
      bars -> scanner -> snapshots -> features -> EVEngine -> decisions

    Assumes an external producer (Step 5) is feeding bar_queue with list[dict] batches,
    where each dict has at least: timestamp, symbol, open, high, low, close, volume
    """
    def __init__(self, *, cfg: LiveRuntimeConfig, bar_queue: "asyncio.Queue[list[dict]]") -> None:
        self.cfg = cfg
        self.bar_queue = bar_queue

        ensure_dir(self.cfg.out_dir)

        self.w_snap = LiveParquetWriter(self.cfg.out_dir, "snapshots", flush_every_s=2.0)
        self.w_dec = LiveParquetWriter(self.cfg.out_dir, "decisions", flush_every_s=2.0)
        self.w_lat = LiveParquetWriter(self.cfg.out_dir, "latency_report", flush_every_s=2.0)

        self.snap_queue: "asyncio.Queue[Dict[str, Any]]" = asyncio.Queue(maxsize=self.cfg.max_snap_queue)

        if self.cfg.ev_root is None:
            raise ValueError("LiveRuntimeConfig.ev_root must be set (folder for EVEngine.from_artifacts).")
        if self.cfg.fe_root is None:
            raise ValueError("LiveRuntimeConfig.fe_root must be set (folder for CoreFeaturePipeline).")

        self.ev = EVEngine.from_artifacts(str(self.cfg.ev_root))
        self.fe = CoreFeaturePipeline(self.cfg.fe_root)

        write_run_manifest(
            self.cfg.out_dir,
            {
                "mode": "live_decisions_only",
                "refresh_s": float(self.cfg.refresh_s),
                "out_dir": str(self.cfg.out_dir),
                "ev_root": str(self.cfg.ev_root),
                "fe_root": str(self.cfg.fe_root),
                "detectors_yaml": self.cfg.detectors_yaml,
                "dev_loose_detectors": bool(self.cfg.dev_loose_detectors),
            },
        )

    async def _decision_loop(self) -> None:
        pca_cols: Optional[list[str]] = None
        batch: list[Dict[str, Any]] = []
        batch_min: Optional[pd.Timestamp] = None

        async def flush(rows: list[Dict[str, Any]]) -> None:
            nonlocal pca_cols
            if not rows:
                return

            snap_df = pd.DataFrame(rows)
            if "timestamp" not in snap_df.columns or "symbol" not in snap_df.columns:
                return

            feats = self.fe.transform_mem(snap_df)
            if feats is None or feats.empty:
                return

            if pca_cols is None:
                pca_cols = [c for c in feats.columns if str(c).startswith("pca_")]
                if not pca_cols:
                    raise RuntimeError("FE produced no pca_* columns; cannot score with vectorize_minute_batch.")

            scored = vectorize_minute_batch(self.ev, feats, pca_cols).frame.copy()
            scored["decision_ts_utc"] = pd.Timestamp.utcnow()

            # a minimal decision flag (you can refine later)
            scored["decision"] = (scored["p_cal"] >= 0.5).astype("int8")

            # latency
            try:
                bar_ts = pd.to_datetime(scored["timestamp"], utc=True, errors="coerce")
                now_ts = pd.Timestamp.utcnow().tz_localize("UTC")
                market_latency_s = (now_ts - bar_ts).dt.total_seconds()
            except Exception:
                market_latency_s = pd.Series([float("nan")] * len(scored))

            lat = pd.DataFrame(
                {
                    "timestamp": scored["timestamp"],
                    "symbol": scored["symbol"],
                    "market_latency_s": market_latency_s.astype("float64"),
                    "snap_queue_depth": float(self.snap_queue.qsize()),
                }
            )

            self.w_dec.append(scored)
            self.w_lat.append(lat)

        while True:
            row = await self.snap_queue.get()
            ts = pd.to_datetime(row.get("timestamp"), utc=True, errors="coerce")
            if pd.isna(ts):
                continue

            ts_min = ts.floor("min")

            if batch_min is None:
                batch_min = ts_min

            if ts_min != batch_min:
                await flush(batch)
                batch = []
                batch_min = ts_min

            batch.append(row)

            # hard safety: avoid unbounded growth even if time bucketing fails
            if len(batch) >= 5000:
                await flush(batch)
                batch = []

    async def run(self, *, stop_after_s: Optional[float] = None) -> None:
        # build detectors from your existing factory
        detectors = build_detectors(self.cfg.detectors_yaml, dev_loose=self.cfg.dev_loose_detectors)

        builder = InProcSnapshotBuilder(self.w_snap, self.snap_queue, max_queue=self.cfg.max_snap_queue)

        scanner = ScannerLoop(
            detectors=detectors,
            builder=builder,
            bar_queue=self.bar_queue,
            refresh_sec=float(self.cfg.refresh_s),  # matches your scanner/live_loop.py signature
        )

        start = time.time()

        scanner.start()
        decision_task = asyncio.create_task(self._decision_loop(), name="decision_loop")

        try:
            while True:
                await asyncio.sleep(1.0)
                if stop_after_s is not None and (time.time() - start) >= float(stop_after_s):
                    break
        finally:
            decision_task.cancel()
            try:
                await decision_task
            except Exception:
                pass
            await scanner.stop()
