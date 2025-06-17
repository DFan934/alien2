# tests/test_csv_pipeline_date_time.py
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import pytest

from data_ingestion.pipelines.csv_pipeline import CSVPipeline


@pytest.fixture(scope="session")
def sample_csv_date_time(tmp_path_factory) -> Path:
    """Create a CSV in 'Date,Time,Open,High,Low,Close,Volume' layout."""
    path = tmp_path_factory.mktemp("data") / "AAPL_2010_Jan.csv"

    t0 = datetime(2010, 1, 12, 9, 30)      # first line of your sample
    rows = []
    for i in range(6):
        ts = t0 + timedelta(minutes=i)
        rows.append(
            dict(
                Date=ts.strftime("%m/%d/%Y"),           # e.g. "01/12/2010"
                Time=ts.strftime("%H:%M"),              # e.g. "09:30"
                Open=54 + i * 0.01,
                High=54 + i * 0.01,
                Low=53.90 + i * 0.01,
                Close=53.98 + i * 0.01,
                Volume=1000 + i * 100,
            )
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_csv_pipeline_date_time(csv_pipeline: CSVPipeline, sample_csv_date_time):
    """Ensure 'Date,Time' layout is parsed correctly."""
    df = csv_pipeline.parse(sample_csv_date_time, symbol="AAPL")

    # Expect 6 rows, timestamp is tz-aware UTC, and strictly monotonic
    assert df.shape[0] == 6
    assert df["timestamp"].dtype == "datetime64[ns, UTC]"
    assert df["timestamp"].is_monotonic_increasing
    assert df["symbol"].unique().tolist() == ["AAPL"]

    # Seconds component should all be zero since we created minute bars
    assert (df["timestamp"].dt.second == 0).all()
