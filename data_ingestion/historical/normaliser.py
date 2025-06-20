############################
# data_ingestion/historical/normaliser.py
############################
"""Normalise raw CSV chunks into canonical schema."""
from __future__ import annotations

import pandas as pd
from datetime import datetime, timezone

REQUIRED_COLS = ["Date", "Time", "Open", "High", "Low", "Close", "Volume"]
OUTPUT_SCHEMA = ["timestamp", "open", "high", "low", "close", "volume", "symbol"]


def clean_chunk(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Return *df* with canonical schema and UTC timestamp."""
    # Ensure required columns exist
    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {symbol} chunk")

    # Combine date + time → timestamp
    ts = pd.to_datetime(df["Date"] + " " + df["Time"], format="%m/%d/%Y %H:%M", utc=True)

    out = pd.DataFrame({
        "timestamp": ts,
        "open": df["Open"].astype("float32"),
        "high": df["High"].astype("float32"),
        "low": df["Low"].astype("float32"),
        "close": df["Close"].astype("float32"),
        "volume": df["Volume"].astype("int32"),
        "symbol": symbol,
    })
    return out