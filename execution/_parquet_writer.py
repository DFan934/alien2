# execution/_parquet_writer.py
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


class AppendParquetDatasetWriter:
    """
    Writes batches to: <out_dir>/<dataset_name>/part-<ts>-<n>.parquet
    Matches your existing test expectations: dataset folder contains part-*.parquet files.
    """
    def __init__(self, out_dir: Path, dataset_name: str) -> None:
        self.dir = Path(out_dir) / dataset_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self._n = 0

    def append_rows(self, rows: List[Dict[str, Any]]) -> Optional[Path]:
        if not rows:
            return None
        df = pd.DataFrame.from_records(rows)
        ts = int(time.time() * 1_000_000)
        self._n += 1
        path = self.dir / f"part-{ts}-{self._n:06d}.parquet"
        df.to_parquet(path, index=False)
        return path
