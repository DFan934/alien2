# ========================
# file: data_ingestion/__init__.py
# ========================
"""Unified data‑ingestion package.

High‑level usage
----------------
>>> from data_ingestion.ingestion_manager import IngestionManager
>>> mgr = IngestionManager.from_yaml('config/ingestion.yaml')
>>> mgr.ingest(symbols=['AAPL', 'MSFT'], timeframe='1Min',
...           start='2020‑01‑01', end='2020‑01‑31')

The package is organised into *connectors* (provider I/O), *pipelines*
(standardisation), *persistence* (Parquet, state), and the
:class:`IngestionManager` orchestrator.
"""

from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import List

__all__: List[str] = [
    "get_connector_cls",
]


_BASE = "data_ingestion.connectors.{}"


def get_connector_cls(name: str):
    """Return the concrete connector class given its short *name* (e.g. ``tradier``).

    Connectors must live in ``data_ingestion.connectors`` and expose a class
    called ``Connector`` that inherits from
    :class:`~data_ingestion.connectors.base.BaseConnector`.
    """
    mod: ModuleType = import_module(_BASE.format(name))
    return getattr(mod, "Connector")
