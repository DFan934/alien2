# data_ingestion/manifest.py
from __future__ import annotations

import json, os, shutil, hashlib, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

MANIFEST_NAME = "manifest.json"
BAD_DIR = "bad"
SCHEMA_VERSION = 1

# ----------------------------- manifest types ------------------------------
@dataclass(frozen=True)
class PartitionEntry:
    symbol: str
    year: int
    month: int
    day: int
    rows: int
    sha256: str
    path: str  # full path to file

@dataclass
class Manifest:
    partitions: List[PartitionEntry]
    schema_version: int = SCHEMA_VERSION
    created_at: str = ""

    @staticmethod
    def _mf_path(root: Path) -> Path:
        return root / MANIFEST_NAME

    @classmethod
    def load(cls, root: Path) -> "Manifest":
        p = cls._mf_path(root)
        if not p.exists():
            return cls(partitions=[], created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        data = json.loads(p.read_text())
        parts = [PartitionEntry(**d) for d in data.get("partitions", [])]
        return cls(partitions=parts,
                   schema_version=data.get("schema_version", SCHEMA_VERSION),
                   created_at=data.get("created_at",""))

    def save(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        obj = {
            "partitions": [asdict(p) for p in self.partitions],
            "schema_version": int(self.schema_version),
            "created_at": self.created_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (self._mf_path(root)).write_text(json.dumps(obj, indent=2, sort_keys=True))

    def append(self, root: Path, entry: PartitionEntry) -> None:
        self.partitions.append(entry)
        self.save(root)

    def rebuild(self, root: Path) -> None:
        """Scan hive tree → manifest (idempotent, append-only semantics preserved by rewriting from scan)."""
        parts: List[PartitionEntry] = []
        for sym_dir in (root).glob("symbol=*"):
            symbol = sym_dir.name.split("=",1)[1]
            for ydir in sym_dir.glob("year=*"):
                year = int(ydir.name.split("=",1)[1])
                for mdir in ydir.glob("month=*"):
                    month = int(mdir.name.split("=",1)[1])
                    for ddir in mdir.glob("day=*"):
                        day = int(ddir.name.split("=",1)[1])
                        for f in ddir.glob("*.parquet"):
                            try:
                                df = pd.read_parquet(f, columns=["timestamp","symbol"])
                                rows = int(len(df))
                                sha = _sha_parquet(df)
                                parts.append(PartitionEntry(symbol, year, month, int(day), rows, sha, str(f)))
                            except Exception:
                                # skip unreadable; user can run quarantine separately
                                continue
        self.partitions = parts
        self.save(root)



# === Phase 2.3: fast, file-system-only partition summary =====================

from datetime import datetime as _dt

def _iter_partition_dirs(root: Path, symbol: str):
    """Yield (year, month, day, dirpath) for existing hive directories; no file IO."""
    sym_root = root / f"symbol={symbol}"
    if not sym_root.exists():
        return
    for ydir in sym_root.glob("year=*"):
        try:
            y = int(ydir.name.split("=", 1)[1])
        except Exception:
            continue
        for mdir in ydir.glob("month=*"):
            try:
                m = int(mdir.name.split("=", 1)[1])
            except Exception:
                continue
            for ddir in mdir.glob("day=*"):
                try:
                    d = int(ddir.name.split("=", 1)[1])
                except Exception:
                    continue
                yield (y, m, d, ddir)

def summarize_partitions_fast(
    root: str | Path,
    symbols: list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> dict:
    """
    Return a JSON-serializable summary WITHOUT reading parquet files.
    Counts *.parquet files by (symbol, year, month, day) within [start, end].
    """
    root = Path(root)
    start = pd.to_datetime(start, utc=True).normalize()
    end   = pd.to_datetime(end,   utc=True).normalize()

    per_symbol = {}
    have_data = 0

    for s in symbols:
        month_counts: dict[tuple[int,int], int] = {}
        total_files = 0

        for (y, m, d, ddir) in _iter_partition_dirs(root, s):
            dt = pd.Timestamp(f"{y:04d}-{m:02d}-{d:02d}", tz="UTC")
            if not (start <= dt <= end):
                continue
            cnt = sum(1 for _ in ddir.glob("*.parquet"))
            if cnt == 0:
                continue
            month_counts[(y, m)] = month_counts.get((y, m), 0) + cnt
            total_files += cnt

        if total_files > 0:
            have_data += 1

        # stable, JSON-friendly shape
        per_symbol[s] = {
            "total_files": int(total_files),
            "by_month": [
                {"year": y, "month": m, "files": int(c)}
                for (y, m), c in sorted(month_counts.items())
            ],
        }

    coverage_ratio = (have_data / max(1, len(symbols)))

    return {
        "symbols": symbols,
        "window": {"start": str(start), "end": str(end)},
        "coverage": {
            "have_data": int(have_data),
            "total": int(len(symbols)),
            "ratio": float(coverage_ratio),
            "empty_symbols": [s for s, rec in per_symbol.items() if rec["total_files"] == 0],
        },
        "summary": per_symbol,
        "generated_at": _dt.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# ------------------------------- helpers -----------------------------------
def _sha_parquet(df: pd.DataFrame) -> str:
    """Row-wise SHA256 over a deterministic subset (timestamp,symbol,open,high,low,close,volume)."""
    cols = ["timestamp","symbol","open","high","low","close","volume"]
    sub = df[cols].copy()
    # freeze to bytes deterministically
    b = sub.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(b).hexdigest()

def _quarantine(root: Path, file_path: Path, reason: str) -> Path:
    rel = file_path.relative_to(root)
    bad_root = root / BAD_DIR / rel.parent
    bad_root.mkdir(parents=True, exist_ok=True)
    bad_path = bad_root / rel.name
    shutil.move(str(file_path), str(bad_path))
    (bad_path.with_suffix(".bad_reason.txt")).write_text(reason)
    return bad_path

# ------------------------------- loader ------------------------------------
@dataclass
class LoadReport:
    symbol: str
    files_read: int
    rows_read: int
    rows_after_concat: int
    rows_after_sort_dedup: int
    rows_dropped_dups: int
    rows_dropped_na: int
    partitions_quarantined: int
    reasons: Dict[str, int]

def _validate_ohlc_block(df: pd.DataFrame) -> bool:
    if df.empty:
        return True
    ok1 = (df["low"] <= df[["open","close"]].min(axis=1)).all()
    ok2 = (df["high"] >= df[["open","close"]].max(axis=1)).all()
    ok3 = (df["high"] >= df["low"]).all()
    return bool(ok1 and ok2 and ok3)

def _validate_monotonic(df: pd.DataFrame) -> bool:
    # per symbol monotonic non-decreasing timestamps
    return bool((df["timestamp"].diff().dropna() >= pd.Timedelta(0)).all())

def load_clean(
    root: Path | str,
    symbol: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    keep_last_on_dup: bool = True,
    row_nan_frac: float = 0.25,
    quarantine_bad: bool = True,
) -> Tuple[pd.DataFrame, LoadReport]:
    """Read filtered hive partitions → concat → sort → de-dup → NA policy → validations."""
    root = Path(root)
    man = Manifest.load(root)

    start = pd.to_datetime(start, utc=True)
    end   = pd.to_datetime(end, utc=True)

    # pick partitions by manifest (fast path); if manifest empty, fall back to scanning
    parts = [p for p in man.partitions if p.symbol == symbol
             and pd.Timestamp(f"{p.year:04d}-{p.month:02d}-{p.day:02d}", tz="UTC") >= start.normalize()
             and pd.Timestamp(f"{p.year:04d}-{p.month:02d}-{p.day:02d}", tz="UTC") <= end.normalize()]
    if not parts:
        # scan on demand (first-time users)
        for ddir in (root / f"symbol={symbol}").glob("year=*/month=*/day=*"):
            y = int(ddir.parent.parent.name.split("=")[1])
            m = int(ddir.parent.name.split("=")[1])
            day = int(ddir.name.split("=")[1])
            dt = pd.Timestamp(f"{y:04d}-{m:02d}-{day:02d}", tz="UTC")
            if start.normalize() <= dt <= end.normalize():
                for f in ddir.glob("*.parquet"):
                    parts.append(PartitionEntry(symbol, y, m, day, rows=0, sha256="", path=str(f)))

    files_read = rows_read = 0
    dfs: List[pd.DataFrame] = []
    quarantined = 0
    reasons: Dict[str,int] = {}

    for p in sorted(parts, key=lambda e: (e.year, e.month, e.day, e.path)):
        fp = Path(p.path)
        if not fp.exists():
            continue
        try:
            df = pd.read_parquet(fp)
            files_read += 1
            rows_read += len(df)

            # basic schema assertions
            req = ["timestamp","open","high","low","close","volume","symbol"]
            missing = set(req) - set(df.columns)
            if missing:
                raise ValueError(f"missing columns {sorted(missing)}")

            # time window trim (defensive)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]

            # OHLC sanity
            if not _validate_ohlc_block(df):
                raise ValueError("OHLC consistency failed")

            dfs.append(df)
        except Exception as e:
            reasons[str(type(e).__name__)] = reasons.get(str(type(e).__name__),0)+1
            if quarantine_bad:
                _quarantine(root, fp, f"load_clean validation failure: {e}")
                quarantined += 1

    if not dfs:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume","symbol"]), LoadReport(
            symbol=symbol, files_read=files_read, rows_read=rows_read, rows_after_concat=0,
            rows_after_sort_dedup=0, rows_dropped_dups=0, rows_dropped_na=0,
            partitions_quarantined=quarantined, reasons=reasons
        )

    df_all = pd.concat(dfs, ignore_index=True)
    rows_after_concat = len(df_all)

    # sort & de-duplicate on (symbol, timestamp)
    df_all = df_all.sort_values(["timestamp","symbol"]).reset_index(drop=True)
    before = len(df_all)
    keep = "last" if keep_last_on_dup else "first"
    df_all = df_all.drop_duplicates(subset=["symbol","timestamp"], keep=keep)
    dups_dropped = before - len(df_all)

    # NA policy: drop rows whose feature NA-fraction > threshold
    # (Safe default: only check core OHLCV here; FE can apply a stricter policy downstream)
    core = ["open","high","low","close","volume"]
    frac = df_all[core].isna().mean(axis=1)
    mask = frac <= float(row_nan_frac)
    na_dropped = int((~mask).sum())
    df_all = df_all.loc[mask].copy()

    # final monotonicity check (post dedup)
    if not _validate_monotonic(df_all):
        # if somehow broken, enforce monotonic order and keep last
        df_all = (df_all
                  .sort_values(["symbol","timestamp"])
                  .drop_duplicates(subset=["symbol","timestamp"], keep="last"))

    rep = LoadReport(
        symbol=symbol,
        files_read=files_read,
        rows_read=rows_read,
        rows_after_concat=rows_after_concat,
        rows_after_sort_dedup=len(df_all),
        rows_dropped_dups=int(dups_dropped),
        rows_dropped_na=int(na_dropped),
        partitions_quarantined=quarantined,
        reasons=reasons,
    )
    return df_all.reset_index(drop=True), rep
