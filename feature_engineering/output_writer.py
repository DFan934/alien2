############################
# feature_engineering/output_writer.py
############################
from __future__ import annotations
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

_PARTITION_COLS = ["symbol", "year", "month"]

def write_feature_dataset(df: pd.DataFrame, out_root: Path) -> None:
    """Write *df* to hive-style Parquet dataset under *out_root*."""
    out_root.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df["year"] = ts.dt.year.astype("int16")
    df["month"] = ts.dt.month.astype("int8")

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_to_dataset(
        table,
        root_path=str(out_root),
        partition_cols=_PARTITION_COLS,
        compression="snappy",
        existing_data_behavior="overwrite_or_ignore",
    )
