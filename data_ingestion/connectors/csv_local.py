# ========================
# file: data_ingestion/connectors/csv_local.py
# ========================
"""Connector that reads minute‑bar CSV files from a local folder.

Assumes Kibot / AlgoSeek‑style filenames: ``<SYMBOL>.csv`` or nested under
symbol directory.  This is primarily for *historical back‑fill*.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd

from .base import BaseConnector

logger = logging.getLogger(__name__)


class Connector(BaseConnector):
    NAME = "csv_local"
    TIMEFRAMES = {"1Min"}

    def __init__(self, root: str | Path, **kwargs):
        super().__init__(root=root, **kwargs)
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(self.root)

    # ------------------------------------------------------------------
    # Implementation
    # ------------------------------------------------------------------

    async def fetch_data(
        self,
        symbols: Sequence[str],
        timeframe: str,
        start_date: str | datetime,
        end_date: str | datetime,
    ) -> pd.DataFrame:
        if timeframe != "1Min":  # only minute bars here
            raise ValueError("csv_local supports only 1Min timeframe")

        dfs: list[pd.DataFrame] = []
        for sym in symbols:
            path = self._resolve_path(sym)
            if not path.exists():
                logger.warning("Missing CSV for %s", sym)
                continue
            df = pd.read_csv(path, parse_dates=[["date", "time"]])
            # normalise column names; keep provider‑native for now
            df.rename(columns={"date_time": "timestamp"}, inplace=True)
            df["symbol"] = sym
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, symbol: str) -> Path:
        # try <root>/<symbol>.csv else <root>/<symbol>/<symbol>.csv
        p1 = self.root / f"{symbol}.csv"
        if p1.exists():
            return p1
        return self.root / symbol / f"{symbol}.csv"


