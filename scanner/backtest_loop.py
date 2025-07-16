# ---------------------------------------------------------------------------
# FILE: scanner/backtest_loop.py  (REPLACED)
# ---------------------------------------------------------------------------
"""Iterates over a historical OHLCV DataFrame and yields `(ts, symbol, snap)`
triples that match exactly what the live scanner would emit – but using the new
`DataGroupBuilder.log_sync()` to avoid any asyncio dependency in back‑tests.
"""
from __future__ import annotations

from typing import Iterator, Tuple

import pandas as pd

from .detectors import CompositeDetector
from .recorder import DataGroupBuilder
from .utils import time_align_minute
from scanner.schema import FEATURE_ORDER

__all__ = ["BacktestScannerLoop"]


class BacktestScannerLoop:
    def __init__(
        self,
        detectors: CompositeDetector,
        builder: DataGroupBuilder,
        df: pd.DataFrame,
    ) -> None:
        self.detectors = detectors
        self.builder = builder
        df = df.copy()
        df.index = df.index.map(time_align_minute)
        df.sort_index(inplace=True)
        self.df = df

    # ----------------------------------------------
    def __iter__(self) -> Iterator[Tuple[pd.Timestamp, str, pd.Series]]:
        for ts, slice_df in self.df.groupby(self.df.index):
            mask = self.detectors(slice_df)
            if not mask.any():
                continue
            for _, row in slice_df[mask].iterrows():
                # Enforce canonical feature order for downstream pipelines
                row = row.loc[list(FEATURE_ORDER)]
                sym = row["symbol"]
                self.builder.log_sync(ts, sym, row)  # <-- sync write
                yield ts, sym, row
