############################
# data_ingestion/historical/ingest_historical.py
############################
"""High‑level orchestrator to convert raw CSVs → Parquet."""
from __future__ import annotations

"""Entry‑point for historical CSV → Parquet conversion."""

import os
import sys
import time
import logging
import pathlib
from typing import Union

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path when run "as a script"
# ---------------------------------------------------------------------------
CUR_DIR = pathlib.Path(__file__).resolve().parent  # .../data_ingestion/historical
PROJ_ROOT = CUR_DIR.parent.parent                  # one level up
if str(PROJ_ROOT) not in sys.path:
    sys.path.append(str(PROJ_ROOT))

# ---------------------------------------------------------------------------
# Local imports (absolute so they work in both run modes)
# ---------------------------------------------------------------------------
from data_ingestion.utils import logger, timeit
from data_ingestion.historical.reader import csv_chunk_generator
from data_ingestion.historical.normaliser import clean_chunk
from data_ingestion.historical.parquet_writer import write_partition

# ---------------------------------------------------------------------------
class HistoricalIngestor:
    """Walks through raw CSVs and writes partitioned Parquet."""

    def __init__(
        self,
        raw_dir: Union[str, os.PathLike] = "raw_data",
        parquet_dir: Union[str, os.PathLike] = "parquet",
        chunk_size: int = 1_000_000,
    ) -> None:
        # Resolve paths relative to *project* root, not CWD
        self.raw_dir = (PROJ_ROOT / pathlib.Path(raw_dir)).resolve()
        self.parquet_dir = (PROJ_ROOT / pathlib.Path(parquet_dir)).resolve()
        self.chunk_size = chunk_size
        self.parquet_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    @timeit("ingestion‑run")
    def run(self) -> None:
        """Main loop: CSV → cleaned chunk → Parquet partition."""
        csv_files = list(self.raw_dir.glob("*.csv"))
        if not csv_files:
            logger.error("No CSVs found under %s", self.raw_dir)
            return

        total_size_gb = sum(f.stat().st_size for f in csv_files) / 1e9
        logger.info("Discovered %d CSV files (%.1f GB) in %s",
                    len(csv_files), total_size_gb, self.raw_dir)

        for csv_path in csv_files:
            symbol = csv_path.stem.upper()
            row_count = 0
            logger.info("Ingesting %s…", csv_path.name)

            for chunk in csv_chunk_generator(csv_path, self.chunk_size):
                chunk = clean_chunk(chunk, symbol)
                write_partition(chunk, self.parquet_dir)
                row_count += len(chunk)

            logger.info("Finished %s → %d rows", symbol, row_count)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    HistoricalIngestor().run()
