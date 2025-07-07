############################
# data_ingestion/historical/normaliser.py
############################
"""Normalise raw CSV chunks and **adjust for corporate actions**."""
from __future__ import annotations

import pandas as pd
from datetime import timezone
from pathlib import Path

from data_ingestion.utils import load_config, logger

REQUIRED_COLS = [
    "Date",
    "Time",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]
OUTPUT_COLS = ["timestamp", "open", "high", "low", "close", "volume", "symbol"]


class CorporateActionsHandler:
    """Loads split / dividend tables and applies *split‑factor* adjustment."""

    def __init__(self) -> None:
        cfg = load_config()
        self._splits = self._load_factors(Path(cfg["splits_path"]), "split_factor")
        self._divs = self._load_factors(Path(cfg["dividends_path"]), "dividend")

    # ------------------------------------------------------------------
    def _load_factors(self, csv_path: Path, col: str) -> pd.DataFrame:
        if not csv_path.exists():
            logger.warning("Corporate‑action file missing: %s – skipping", csv_path)
            return pd.DataFrame(columns=["symbol", "date", col])

        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.normalize()
        df[col] = pd.to_numeric(df[col])
        return df

    # ------------------------------------------------------------------
    def adjust(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return *df* adjusted for splits.  Dividend handling = passthrough (price
        not adjusted but column available if you need)."""
        if df.empty:
            return df

        sym = df["symbol"].iat[0]
        if sym not in self._splits.symbol.unique():
            return df  # nothing to do

        splits = self._splits[self._splits.symbol == sym]
        if splits.empty:
            return df

        # Build cumulative adjustment factor (earliest row first)
        splits = splits.sort_values("date")
        splits["cum_factor"] = splits["split_factor"].cumprod()

        # Merge on date
        df = df.copy()
        df["date"] = df["timestamp"].dt.normalize()
        df = df.merge(splits[["date", "cum_factor"]], on="date", how="left")
        df["cum_factor"].fillna(method="ffill", inplace=True)
        df["cum_factor"].fillna(1.0, inplace=True)

        adjust_cols = ["open", "high", "low", "close"]
        df.loc[:, adjust_cols] = df[adjust_cols].div(df["cum_factor"], axis=0)
        df.drop(columns=["cum_factor", "date"], inplace=True)
        return df


# Global singleton – cheap
_CA_HANDLER = CorporateActionsHandler()


def clean_chunk(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Return *df* with canonical schema, UTC timestamp, **split‑adjusted** prices."""
    # --- schema validation -------------------------------------------------
    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {symbol} chunk")

    # --- date parsing ------------------------------------------------------
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

    # --- corporate actions -------------------------------------------------
    out = _CA_HANDLER.adjust(out)
    return out