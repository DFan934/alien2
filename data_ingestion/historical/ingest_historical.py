############################
# data_ingestion/historical/ingest_historical.py
############################
"""High‑level orchestrator – now config‑driven + mem‑aware."""
from __future__ import annotations

import sys, time, pathlib
from typing import Union

from data_ingestion.utils import logger, timeit, load_config
from data_ingestion.historical.reader import csv_chunk_generator
from data_ingestion.historical.normaliser import clean_chunk
from data_ingestion.historical.parquet_writer import write_partition

CUR_DIR = pathlib.Path(__file__).resolve().parent
PROJ_ROOT = CUR_DIR.parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.append(str(PROJ_ROOT))

_cfg = load_config()


class HistoricalIngestor:
    """Convert raw CSV minute bars → split‑adjusted Parquet dataset."""

    def __init__(
        self,
        raw_dir: Union[str, pathlib.Path] = "raw_data",
        parquet_dir: Union[str, pathlib.Path] = "parquet",
    ) -> None:
        self.raw_dir = (PROJ_ROOT / pathlib.Path(raw_dir)).resolve()
        self.parquet_dir = (PROJ_ROOT / pathlib.Path(parquet_dir)).resolve()
        self.parquet_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    @timeit("ingestion‑run")
    def run(self) -> None:
        csv_files = list(self.raw_dir.glob("*.csv"))
        if not csv_files:
            logger.error("No CSV files under %s", self.raw_dir)
            return

        logger.info("Found %d CSVs in %s", len(csv_files), self.raw_dir)

        for csv_path in csv_files:
            symbol = csv_path.stem.upper()
            logger.info("Ingesting %s…", symbol)
            t0 = time.perf_counter()
            rows = 0
            for chunk in csv_chunk_generator(csv_path):
                chunk = clean_chunk(chunk, symbol)
                write_partition(chunk, self.parquet_dir)
                rows += len(chunk)
            logger.info("%s → %d rows (%.1fs)", symbol, rows, time.perf_counter() - t0)


if __name__ == "__main__":
    HistoricalIngestor().run()
