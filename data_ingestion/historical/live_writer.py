# artifacts/live_writer.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class LiveParquetWriter:
    """
    Append-only Parquet writer implemented as a directory of part files:
        <root>/<name>.parquet/part-000001.parquet
    This avoids "true parquet append" complexity and stays stable for long runs.
    """
    out_dir: Path
    name: str
    flush_every_s: float = 2.0

    def __post_init__(self) -> None:
        self.dataset_dir = self.out_dir / f"{self.name}.parquet"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self._buf: list[pd.DataFrame] = []
        self._last_flush = 0.0
        self._part = self._load_part_counter()

    def _counter_path(self) -> Path:
        return self.dataset_dir / "_counter.json"

    def _load_part_counter(self) -> int:
        p = self._counter_path()
        if p.exists():
            try:
                return int(json.loads(p.read_text()).get("next_part", 1))
            except Exception:
                return 1
        return 1

    def _save_part_counter(self) -> None:
        self._counter_path().write_text(json.dumps({"next_part": self._part}, indent=2))

    def append(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            return
        self._buf.append(df)

        now = time.time()
        if (now - self._last_flush) >= self.flush_every_s:
            self.flush()

    def flush(self) -> Optional[Path]:
        if not self._buf:
            return None

        df = pd.concat(self._buf, axis=0, ignore_index=True)
        self._buf.clear()

        part_path = self.dataset_dir / f"part-{self._part:06d}.parquet"
        self._part += 1
        self._save_part_counter()

        df.to_parquet(part_path, index=False)
        self._last_flush = time.time()
        return part_path


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def now_run_id() -> str:
    # local time string is fine for run folder naming
    return time.strftime("%Y%m%d_%H%M%S")


def write_run_manifest(out_dir: Path, manifest: Dict[str, Any]) -> None:
    (out_dir / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
