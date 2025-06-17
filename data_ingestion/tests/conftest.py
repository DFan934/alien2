# tests/conftest.py
import json
import os
from pathlib import Path
import pandas as pd
import pytest
from datetime import datetime, timedelta
from data_ingestion.pipelines.csv_pipeline import CSVPipeline
from data_ingestion.ingestion_manager import IngestionManager

@pytest.fixture(scope="session")
def sample_csv(tmp_path_factory) -> Path:
    """Generate a tiny 1-minute-bar CSV for symbol SPY."""
    path = tmp_path_factory.mktemp("data") / "SPY_202501.csv"
    t0 = datetime(2025, 1, 2, 9, 30)          # first US session of the year
    rows = []
    for i in range(5):                         # five minutes is plenty for unit tests
        ts = t0 + timedelta(minutes=i)
        rows.append(
            dict(
                Timestamp=ts.strftime("%Y-%m-%d %H:%M:%S"),
                Open=470.0 + i,
                High=470.5 + i,
                Low=469.5 + i,
                Close=470.2 + i,
                Volume=1_000 + i * 10,
            )
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def csv_pipeline():
    return CSVPipeline()


@pytest.fixture(scope="function")
def tmp_state_file(tmp_path):
    """Return a file path that tests can use for breaker-state persistence."""
    return tmp_path / "ing_state.json"


@pytest.fixture(scope="function")
def ingestion_manager(tmp_state_file):
    return IngestionManager(state_path=tmp_state_file)
