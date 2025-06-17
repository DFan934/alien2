# ========================
# file: data_ingestion/pipelines/csv_pipeline.py
# ========================
"""Standard-isation pipeline: raw minute-bar CSV → canonical bar frame.

Canonical schema:
    timestamp (UTC, ns), symbol, open, high, low, close, volume
"""

from __future__ import annotations

import logging
from datetime import timezone
from pathlib import Path

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Data‐frame schema (basic) – extend later with range checks, etc.
# ---------------------------------------------------------------------
_RAW_SCHEMA = DataFrameSchema(
    {
        "timestamp": Column(pa.Timestamp, nullable=False),
        "open": Column(float, nullable=False),
        "high": Column(float, nullable=False),
        "low": Column(float, nullable=False),
        "close": Column(float, nullable=False),
        "volume": Column(float, nullable=False),
        "symbol": Column(str, nullable=False),
    },
    coerce=True,
)


class CSVPipeline:
    """Parse a *single* raw CSV (Kibot, Polygon, etc.) into canonical bars.

    Parameters
    ----------
    path : pathlib.Path
        Location of the CSV file.
    symbol : str
        Ticker represented by the file (e.g. ``"SPY"``).

    Returns
    -------
    pandas.DataFrame
        Canonical, validated minute bars sorted by ``timestamp``.
    """

    REQUIRED = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "symbol",
    ]

    # -----------------------------------------------------------------
    # Public API – used by tests and ingestion manager
    # -----------------------------------------------------------------
    def parse(self, path: Path, *, symbol: str) -> pd.DataFrame:
        """Load CSV, stamp ticker, standardise columns, and validate."""
        logger.debug("Parsing %s for symbol=%s", path, symbol)

        raw = self._read_csv_auto(path)
        raw["symbol"] = symbol.upper()
        raw["symbol"] = raw["symbol"].astype("category")  # match Parquet round-trip

        df = self._standardise(raw)
        return df[self.REQUIRED]

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------
    @staticmethod
    def _read_csv_auto(path: Path) -> pd.DataFrame:
        """Handle both ‘Date,Time,…’ and single ‘Timestamp’ styles."""
        df = pd.read_csv(path)

        # Normalise column names to lower-case once up front
        df.columns = [c.lower() for c in df.columns]

        if {"date", "time"}.issubset(df.columns):
            # Example: 01/12/2010,09:30 → 2010-01-12 09:30:00-05:00
            df["timestamp"] = pd.to_datetime(
                df["date"] + " " + df["time"],
                format="%m/%d/%Y %H:%M",
                utc=True,
                errors="coerce",
            )
            df.drop(columns=["date", "time"], inplace=True)

        elif "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

        else:
            raise ValueError(
                f"{path.name} lacks expected ('date'+'time') or 'timestamp' columns."
            )

        # Lower-case already done; return for further standardisation
        return df

    def _standardise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate schema, ensure UTC tz-aware index, sort, drop NA rows."""
        df = _RAW_SCHEMA.validate(df, lazy=True)

        # enforce UTC, drop un-parseable rows
        df["timestamp"] = (
            pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            .dt.tz_convert(timezone.utc)
        )
        df.dropna(subset=["timestamp"], inplace=True)

        # canonical order
        df.sort_values(["symbol", "timestamp"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df
