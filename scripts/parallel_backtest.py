# ---------------------------------------------------------------------------
# Parallel scanner replay – zero-CLI version
# ---------------------------------------------------------------------------
from __future__ import annotations
import logging, multiprocessing as mp
from pathlib import Path
from typing import Dict, List

import dask.dataframe as dd
import pandas as pd
from universes import resolve_universe, UniverseError
import universes.providers as U
from pathlib import Path

from scanner.detectors import GapDetector, HighRVOLDetector, CompositeDetector
from scanner.backtest_loop import BacktestScannerLoop
from scanner.recorder import DataGroupBuilder
from scanner.utils import time_align_minute

# ────────────────────────────────────────────────────────────────────────
# CONFIG  –  EDIT HERE ONCE, THEN  python scripts2/parallel_backtest.py
# ────────────────────────────────────────────────────────────────────────
CONFIG: Dict[str, object] = {
    # Raw hive-partitioned minute parquet dataset (symbol=…/year=…/month=…)
    "parquet_root": "parquet/minute_bars",
    #"symbols": ["RRC"],
    "universe": ["RRC"],  # OR "path/to/universe.csv" OR {"type":"sp500","as_of":"YYYY-MM-DD"}
    "universe_max_size": 5000,

    "start": "1998-08-26",
    "end":   "1998-11-26",

    # Detector thresholds
    "gap_pct": 0.02,
    "rvol":    2.0,

    # Where snapshots land (same layout as live scanner)
    "out_dir": "scanner_events_parallel",
}
# ────────────────────────────────────────────────────────────────────────


# ---------------------------------------------------------------------------
# Partition worker
# ---------------------------------------------------------------------------
def _run_partition(pdf: pd.DataFrame, cfg: Dict[str, object]) -> int:
    """Executed inside a Dask worker process on one Pandas partition."""
    detectors = CompositeDetector([
        GapDetector(pct=float(cfg["gap_pct"])),
        HighRVOLDetector(thresh=float(cfg["rvol"])),
    ])
    builder = DataGroupBuilder(cfg["out_dir"])      # buffer inside each worker
    loop = BacktestScannerLoop(detectors, builder, pdf)
    n = sum(1 for _ in loop)                        # iterate & persist
    builder.flush()                                 # make sure buffer hits disk
    return n


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def main(cfg: Dict[str, object]) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("parallel_backtest")

    root     = Path(cfg["parquet_root"])
    #symbols  = cfg["symbols"]

    try:
        #symbols = resolve_universe(cfg, as_of=cfg.get("as_of"))
        symbols = U.resolve_universe(cfg, as_of=cfg.get("as_of"))

    except NotImplementedError as e:
        raise RuntimeError("SP500Universe(as_of=...) not implemented in Phase 2. Use Static/File universe.") from e
    except UniverseError as e:
        raise RuntimeError(f"Invalid universe in CONFIG: {e}") from e

    print(f"[Universe] size = {len(symbols)} | {symbols}")

    start, end = cfg["start"], cfg["end"]

    # ---- Phase-2.2 acceptance: bail early if no parquet exists (test expects RuntimeError)
    parquet_root = Path(cfg.get("parquet_root", "parquet"))
    # If there are no parquet files at all, raise before Dask tries to scan
    if not parquet_root.exists() or not any(parquet_root.rglob("*.parquet")):
        raise RuntimeError(f"No parquet files found under {parquet_root!s}")

    # 1) lazy-load slice with Dask ------------------------------------------------
    log.info("Loading parquet slice via Dask …")
    ddf = dd.read_parquet(
        root,
        filters=[
            ("symbol", "in", symbols),
            ("timestamp", ">=", start),
            ("timestamp", "<=", end),
        ],
        engine="pyarrow",
        gather_statistics=False,
    )
    # Snap every timestamp to minute bucket (deterministic)
    ddf["timestamp"] = dd.to_datetime(ddf["timestamp"])
    ddf = ddf.map_partitions(
        lambda pdf: pdf.assign(timestamp=pdf["timestamp"].map(time_align_minute))
    )

    # 2) Run scanner in parallel -------------------------------------------------
    log.info("Spinning up %d CPU workers over %d Dask partitions …",
             mp.cpu_count(), ddf.npartitions)
    total = ddf.map_partitions(_run_partition, cfg).compute(scheduler="processes")
    log.info("Done. Scanner events recorded: %d", int(total.sum()))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main(CONFIG)