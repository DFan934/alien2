############################
# tests/test_ingestion.py
############################
"""Basic sanity tests for ingestion pipeline."""
import pandas as pd
from data_ingestion.historical.normaliser import clean_chunk


def test_clean_chunk_schema():
    df = pd.DataFrame({
        "Date": ["01/12/2010"],
        "Time": ["09:30"],
        "Open": [54.0],
        "High": [54.0],
        "Low": [53.9],
        "Close": [53.98],
        "Volume": [17065],
    })
    out = clean_chunk(df, "AAPL")
    assert list(out.columns) == ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
