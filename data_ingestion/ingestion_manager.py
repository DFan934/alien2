# ========================
# file: data_ingestion/ingestion_manager.py
# ========================
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict

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
    def fetch_bars(self, *args, **kwargs):
        """Main entry – choose an available connector and fetch data."""
        ...

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
