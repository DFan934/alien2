############################
# data_ingestion/historical/reader.py
############################
"""Chunked CSV reader for large minute‑bar files."""
from __future__ import annotations

import os, pathlib, pandas as pd, math
from data_ingestion.utils import load_config, logger

_cfg = load_config()
_CHUNK_MEM_MB = int(_cfg.get("chunk_mem_target_mb", 256))
_ON_BAD = _cfg.get("on_bad_lines", "skip")

# Approx bytes per row for dtype mix (empirical ≈ 56 B) – tweak if needed
_BYTES_PER_ROW = 56
_CHUNK_ROWS = max(1, (_CHUNK_MEM_MB * 1_048_576) // _BYTES_PER_ROW)
logger.info("Reader chunk_rows set to %d (≈%d MB target)", _CHUNK_ROWS, _CHUNK_MEM_MB)


def csv_chunk_generator(csv_path: str | os.PathLike, chunk_rows: int | None = None):
    """Yield DataFrame *chunks* of ≈ ``chunk_rows`` rows (auto‑sized by mem target)"""
    dtype_map = {
        "Open": "float32",
        "High": "float32",
        "Low": "float32",
        "Close": "float32",
        "Volume": "int32",
    }
    rows = chunk_rows or _CHUNK_ROWS
    try:
        for chunk in pd.read_csv(
            csv_path,
            chunksize=rows,
            dtype=dtype_map,
            parse_dates=False,
            on_bad_lines=_ON_BAD,
        ):
            yield chunk
    except UnicodeDecodeError as e:
        logger.error("Unicode error in %s – %s", csv_path, e)
        return