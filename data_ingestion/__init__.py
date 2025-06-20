############################
# data_ingestion/__init__.py
############################
"""High‑level import convenience."""
from pathlib import Path

from .historical.ingest_historical import HistoricalIngestor  # noqa: F401

# Expose package version for downstream logging
__version__ = "0.1.0"

