############################
# data_ingestion/historical/parquet_writer.py
############################
"""Write cleaned chunks to partitioned Parquet dataset."""
from __future__ import annotations

import pyarrow as pa, pyarrow.parquet as pq, pyarrow.dataset as ds
import os, pathlib, pandas as pd
from data_ingestion.utils import logger


TABLE_SCHEMA = pa.schema([
    ("timestamp", pa.timestamp("us", tz="UTC")),
    ("open", pa.float32()),
    ("high", pa.float32()),
    ("low", pa.float32()),
    ("close", pa.float32()),
    ("volume", pa.int32()),
    ("symbol", pa.string()),
])


def write_partition(df: pd.DataFrame, parquet_root: pathlib.Path):
    """Append *df* to symbol/year/month hive partition with snappy compression."""
    if df.empty:
        return

    symbol = df["symbol"].iat[0]
    year = df["timestamp"].dt.year.iloc[0]
    month = f"{df['timestamp'].dt.month.iloc[0]:02d}"

    path = parquet_root / f"symbol={symbol}" / f"year={year}" / f"month={month}"
    path.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(df, schema=TABLE_SCHEMA, preserve_index=False)
    file_path = path / f"{symbol}_{year}_{month}.parquet"
    pq.write_table(table, file_path, compression="snappy", use_deprecated_int96_timestamps=True)
    logger.debug(f"Wrote {len(df):,} rows → {file_path}")
