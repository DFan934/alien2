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
        # Iterate per symbol so detectors see a real time-series (not a 1-minute cross-section).
        for sym, df_sym in self.df.groupby("symbol", sort=False):
            # --- diagnostics: INPUT (per symbol) ---
            try:
                print(
                    f"[Scanner-INPUT] {sym} "
                    f"rows={len(df_sym)} "
                    f"ts_min={df_sym.index.min()} "
                    f"ts_max={df_sym.index.max()}"
                )
            except Exception as e:
                print(f"[Scanner-INPUT] {sym} (failed to print) err={type(e).__name__}: {e}")

            mask = self.detectors(df_sym)
            mask = pd.Series(mask, index=df_sym.index).astype(bool)

            df_kept = df_sym.loc[mask]

            # --- diagnostics: OUTPUT (per symbol) ---
            print(
                f"[Scanner-OUTPUT] {sym} "
                f"rows={len(df_kept)} "
                f"kept_pct={len(df_kept) / max(len(df_sym), 1):.2%}"
            )

            if df_kept.empty:
                continue

            for ts, row in df_kept.iterrows():
                # enforce schema order (same code you already have)
                missing = [c for c in FEATURE_ORDER if c not in row.index]
                if missing:
                    raise KeyError(
                        "Scanner snapshot schema mismatch: row is missing required columns: "
                        + ", ".join(missing[:25])
                        + (f" ... (+{len(missing) - 25} more)" if len(missing) > 25 else "")
                    )
                row = row.reindex(list(FEATURE_ORDER))
                self.builder.log_sync(ts, sym, row)
                yield ts, sym, row

