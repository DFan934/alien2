############################
# data_ingestion/historical/parquet_writer.py
############################
"""Write cleaned chunks to hive‑partitioned Parquet **with schema lock‑in**."""
from __future__ import annotations
import hashlib

import json
import pathlib
from typing import Dict, Optional
import os, uuid, tempfile, time
from pathlib import Path
from data_ingestion.manifest import Manifest, PartitionEntry, _sha_parquet

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy.testing.plugin.plugin_base import logging

from data_ingestion.utils import logger

TABLE_SCHEMA = pa.schema(
    [
        ("timestamp", pa.timestamp("us", tz="UTC")),
        ("open", pa.float32()),
        ("high", pa.float32()),
        ("low", pa.float32()),
        ("close", pa.float32()),
        ("volume", pa.int32()),
        ("symbol", pa.string()),
    ]
)

_SCHEMA_FILE = "schema.json"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _schema_path(parquet_root: pathlib.Path) -> pathlib.Path:
    return parquet_root / _SCHEMA_FILE


def _save_schema(parquet_root: pathlib.Path) -> None:
    """Serialise TABLE_SCHEMA -> schema.json (one-time)."""
    with open(_schema_path(parquet_root), "w", encoding="utf-8") as fh:
        json.dump({f.name: str(f.type) for f in TABLE_SCHEMA}, fh, indent=2)


def _load_schema(parquet_root: pathlib.Path) -> Optional[Dict[str, str]]:
    """Return the saved schema mapping or None if it does not exist / is invalid."""
    p = _schema_path(parquet_root)
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError:
        logger.warning("Corrupt schema.json detected – regenerating.")
        return None

def _normalize_pandas_dtype(dtype_str: str) -> str:
    s = str(dtype_str)
    # normalize floats
    if s in ("float32", "float64"):
        return "float"  # matches PyArrow's str(pa.float32()) == "float"
    # normalize ints (we only expect int32 in manifest, but don't die on int64)
    if s in ("int32", "int64"):
        return "int32"
    # normalize strings
    if s in ("object", "string"):
        return "string"
    # normalize timestamps: accept ns/us tz-aware as 'timestamp[us, tz=UTC]'
    if s.startswith("datetime64[") and "UTC" in s:
        return "timestamp[us, tz=UTC]"
    return s


def _validate_schema(df: pd.DataFrame, parquet_root: pathlib.Path) -> None:
    """Ensure df dtypes match the manifest; create it on first run."""
    saved = _load_schema(parquet_root)
    if saved is None:
        _save_schema(parquet_root)
        logger.info("Saved Parquet schema manifest → %s", _schema_path(parquet_root))
        return

    current = {c: _normalize_pandas_dtype(str(t)) for c, t in zip(df.columns, df.dtypes)}
    # Also normalize saved in case different PyArrow versions stringify differently
    saved_norm = {k: _normalize_pandas_dtype(v) for k, v in saved.items()}

    mismatch = {
        c: (current.get(c, "<missing>"), saved_norm[c])
        for c in saved_norm
        if saved_norm[c] != current.get(c)
    }
    if mismatch:
        raise TypeError(f"Schema mismatch vs. manifest: {mismatch}")


def _atomic_write_table(table: pa.Table, final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=".tmp_", dir=str(final_path.parent), delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        pq.write_table(
            table,
            tmp_path,
            compression="snappy",
            use_deprecated_int96_timestamps=True,
        )
        os.replace(tmp_path, final_path)  # atomic on same filesystem
    finally:
        if tmp_path.exists():
            try: tmp_path.unlink()
            except Exception: pass



# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
'''def write_partition(df: pd.DataFrame, parquet_root: pathlib.Path) -> None:
    """Append *df* to symbol/year/month hive partition with Snappy compression."""
    if df.empty:
        return

    _validate_schema(df, parquet_root)

    import logging
    _log = logging.getLogger(__name__)
    if _log.isEnabledFor(logging.DEBUG):
    #if logger.isEnabledFor(logging.DEBUG):
        col_sha = hashlib.sha1(",".join(df.columns).encode()).hexdigest()[:10]
        logger.debug("[WRITE] %s rows=%d  sha=%s → %s",
                     df["symbol"].iat[0], len(df), col_sha, parquet_root)

    symbol = df["symbol"].iat[0]
    ts = df["timestamp"].dt
    year, month = ts.year.iloc[0], f"{ts.month.iloc[0]:02d}"

    path = parquet_root / f"symbol={symbol}" / f"year={year}" / f"month={month}"
    path.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(df, schema=TABLE_SCHEMA, preserve_index=False)
    fname = path / f"{symbol}_{year}_{month}.parquet"
    pq.write_table(
        table,
        fname,
        compression="snappy",
        use_deprecated_int96_timestamps=True,
    )
    logger.debug("Wrote %d rows → %s", len(df), fname)'''


def write_partition(df: pd.DataFrame, parquet_root: pathlib.Path) -> None:
    """Append df into hive partitions (symbol/year/month/day), atomically, with manifest entry."""
    if df.empty:
        return

    _validate_schema(df, parquet_root)

    symbol = str(df["symbol"].iat[0]).upper()
    ts = df["timestamp"].dt

    year  = int(ts.year.iloc[0])
    month = int(ts.month.iloc[0])
    day   = int(ts.day.iloc[0])

    part_dir = Path(parquet_root) / f"symbol={symbol}" / f"year={year:04d}" / f"month={month:02d}" / f"day={day:02d}"
    part_dir.mkdir(parents=True, exist_ok=True)

    # unique, append-only filename (prevents clobber on re-ingest)
    chunk_id = uuid.uuid4().hex[:12]
    fname = f"part-{year:04d}{month:02d}{day:02d}-{chunk_id}.parquet"
    final_path = part_dir / fname

    table = pa.Table.from_pandas(df, schema=TABLE_SCHEMA, preserve_index=False)

    # atomic write
    _atomic_write_table(table, final_path)

    # manifest append
    man = Manifest.load(Path(parquet_root))
    sha = _sha_parquet(df)
    entry = PartitionEntry(
        symbol=symbol, year=year, month=month, day=day,
        rows=int(len(df)), sha256=sha, path=str(final_path)
    )
    man.append(Path(parquet_root), entry)

    logger.debug("Wrote %d rows → %s", len(df), final_path)
