# -----------------------------------------------------------------------------
# 5) scanner/tests/test_snapshot_schema.py
# -----------------------------------------------------------------------------

"""Ensures snapshot schema matches feature pipeline list."""
import json, tempfile, pandas as pd
from pathlib import Path

import pytest

from scanner.recorder import DataGroupBuilder
from scanner.detectors import GapDetector
from scanner.utils import time_align_minute

# Import feature list from core pipeline
from feature_engineering.pipelines.core import CoreFeaturePipeline  # type: ignore

@pytest.mark.asyncio
async def test_snapshot_schema(tmp_path: Path):    # --- 1. fabricate dummy bar & snapshot ----------------------
    now = pd.Timestamp.utcnow().floor("1min")
    df = pd.DataFrame(
        {
            "open": [100.0],
            "close": [102.0],
            "high": [103.0],
            "low": [99.5],
            "volume": [1_000_000],
            "prev_close": [97.0],
            "symbol": ["AAPL"],
        },
        index=[now],
    )

    builder = DataGroupBuilder(parquet_root=tmp_path)
    det = GapDetector(pct=0.02)
    mask = det(df)
    ts = time_align_minute(now.to_pydatetime())
    await builder.log(ts, "AAPL", df.loc[now])

    # --- 2. read written parquet (single row) -------------------
    pq_files = list(tmp_path.rglob("*.parquet"))
    assert pq_files, "No parquet written by DataGroupBuilder"
    snap_df = pd.read_parquet(pq_files[0])

    expected_cols = CoreFeaturePipeline.FEATURE_ORDER  # list[str]
    assert set(snap_df.columns).issuperset(expected_cols), (
        "Missing columns in scanner snapshot: "
        f"{set(expected_cols) - set(snap_df.columns)}"
    )