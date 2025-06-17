# ========================
# file: data_ingestion/schemas.py
# ========================
"""Central place for pandera / pydantic schemas.  (Minimal for now)."""

from __future__ import annotations

import pandera.pandas as pa
from pandera import Column, DataFrameSchema

CANONICAL_BAR = DataFrameSchema(
    {
        "timestamp": Column(pa.Timestamp),
        "symbol": Column(str),
        "open": Column(float),
        "high": Column(float),
        "low": Column(float),
        "close": Column(float),
        "volume": Column(float),
    },
    coerce=True,
)

