############################
# data_ingestion/historical/reader.py
############################
"""Chunked CSV reader for large minute‑bar files."""
from __future__ import annotations

import os, pathlib, pandas as pd


def csv_chunk_generator(csv_path: str | os.PathLike, chunk_size: int = 1_000_000):
    """Yield DataFrame chunks from *csv_path* of size ``chunk_size`` rows."""
    dtype_map = {
        "Open": "float32",
        "High": "float32",
        "Low": "float32",
        "Close": "float32",
        "Volume": "int32",
    }
    for chunk in pd.read_csv(
        csv_path,
        chunksize=chunk_size,
        dtype=dtype_map,
        parse_dates=False,  # raw strings; normaliser will parse
        on_bad_lines="skip",
    ):
        yield chunk