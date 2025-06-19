# ========================
# file: data_ingestion/ingestion_manager.py
# ========================
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict

from datetime import datetime
from data_ingestion.connectors.csv_local import Connector as CSVLocal
from data_ingestion.pipelines.csv_pipeline import CSVPipeline
from data_ingestion.persistence import write_parquet
from .persistence import write_parquet

import pandas as pd
logger = logging.getLogger(__name__)


class IngestionManager:
    """Orchestrates connector calls, circuit-breaker, and persistence."""

    BREAKER_WINDOW = 60 * 5  # 5-minute cooldown after a provider error

    # -----------------------------------------------------------------
    # constructor
    # -----------------------------------------------------------------
    def __init__(
        self,
        *,
        state_path: str | Path | None = None,
        providers: list | None = None,
    ):
        # where to persist breaker state
        self._state_path = Path(state_path) if state_path else Path("ingestion_state.json")
        self.breaker_state: Dict[str, dict] = self._load_state()

        # initialise connectors (stubbed for brevity)
        self.connectors = providers or []
        logger.debug("IngestionManager created with state_path=%s", self._state_path)

    # -----------------------------------------------------------------
    # public API
    # -----------------------------------------------------------------
    async def fetch_bars(
        self,
        symbols: list[str],
        timeframe: str,
        start: datetime | str,
        end: datetime | str,
        out_root: Path = Path("raw/minute"),
    ) -> Path:
        """Return path to partitioned Parquet after assuring data is on disk."""
        # 1) choose provider (only csv_local for now)
        cx = CSVLocal(root=Path("historical_csv"))

        if not self._breaker_window_expired(cx.NAME):
            raise RuntimeError(f"Provider {cx.NAME} still in cooldown")

        try:
            raw = await cx.fetch_data(symbols, timeframe, start, end)
        except Exception as exc:                   # CSV read failure, etc.
            self._trip_breaker(cx.NAME)
            raise

        # 2) pipeline standardisation
        pipe = CSVPipeline()
        frames = [pipe.parse(f, symbol=sym) for sym, f in zip(symbols, raw)]

        df = pd.concat(frames, ignore_index=True)

        # 3) persist with schema tag
        path = write_parquet(df, out_root, partition_cols=["symbol", "year"])
        return path

    # -----------------------------------------------------------------
    # circuit-breaker helpers
    # -----------------------------------------------------------------
    def _trip_breaker(self, provider_name: str):
        logger.warning("Circuit breaker tripped for %s", provider_name)
        now = time.time()
        self.breaker_state[provider_name] = {"last_trip": now}
        self._save_state()

    def _breaker_window_expired(self, provider_name: str) -> bool:
        state = self.breaker_state.get(provider_name)
        if not state:
            return True       # never tripped
        return (time.time() - state["last_trip"]) > self.BREAKER_WINDOW

    # -----------------------------------------------------------------
    # persistence
    # -----------------------------------------------------------------
    def _load_state(self) -> Dict[str, dict]:
        if self._state_path.exists():
            try:
                with self._state_path.open("r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception as exc:  # pragma: no cover
                logger.error("Could not read breaker state – starting fresh (%s)", exc)
        return {}

    def _save_state(self):
        try:
            with self._state_path.open("w", encoding="utf-8") as fh:
                json.dump(self.breaker_state, fh)
        except Exception as exc:      # pragma: no cover
            logger.error("Failed to persist breaker state: %s", exc)
