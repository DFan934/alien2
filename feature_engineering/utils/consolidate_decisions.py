# artifacts/consolidate_decisions.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import json
import pandas as pd


@dataclass(frozen=True)
class ConsolidationResult:
    run_id: str
    artifacts_root: Path
    searched_dirs: Tuple[Path, ...]
    decision_files_found: Tuple[Path, ...]
    consolidated_path: Optional[Path]
    rows_written: int
    unique_source_files: int


def _read_decisions_file(p: Path) -> pd.DataFrame:
    """
    Supported formats:
      - .parquet
      - .csv
      - .jsonl (one JSON object per line)
    """
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix == ".jsonl":
        rows = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return pd.DataFrame(rows)
    raise ValueError(f"Unsupported decisions file type: {p} (suffix={suffix})")


def _dedupe_keep_last(df: pd.DataFrame, key_cols: Sequence[str]) -> pd.DataFrame:
    """
    Dedupe by key columns keeping the last row (assumes later rows are “newer”).
    If key columns aren’t present, returns df unchanged.
    """
    for c in key_cols:
        if c not in df.columns:
            return df
    # stable sort not guaranteed; this is best-effort.
    return df.drop_duplicates(subset=list(key_cols), keep="last")


def _ensure_parent_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def find_decision_files(
    artifacts_root: Path,
    searched_dirs: Iterable[Path],
    patterns: Sequence[str] = (
        "**/decisions.parquet",
        "**/decisions.csv",
        "**/decisions.jsonl",
        "**/*decisions*.parquet",
        "**/*decisions*.csv",
        "**/*decisions*.jsonl",
    ),
) -> List[Path]:
    """
    Search for decision artifacts in the given directories. Patterns are recursive.
    Returns sorted unique list of files that exist.
    """
    found: List[Path] = []
    seen = set()

    for d in searched_dirs:
        d = d.expanduser().resolve()
        if not d.exists():
            continue
        for pat in patterns:
            for p in d.glob(pat):
                try:
                    rp = p.resolve()
                except Exception:
                    rp = p
                if rp.is_file() and rp not in seen:
                    seen.add(rp)
                    found.append(rp)

    # deterministic order
    found.sort(key=lambda x: str(x))
    return found


def consolidate_decisions(
    *,
    run_id: str,
    artifacts_root: Path,
    searched_dirs: Sequence[Path],
    output_relpath: str = "consolidated/decisions_consolidated.parquet",
    dedupe_key_cols: Sequence[str] = ("symbol", "ts", "decision_id"),
    require_columns: Sequence[str] = (),  # optional: enforce schema
    verbose: bool = True,
) -> ConsolidationResult:
    """
    Consolidate all decision files found under searched_dirs into a single parquet file.

    Hard guarantees:
      - rows_written is exact number of rows persisted
      - consolidated_path is None if and only if no decision files were found
      - decision_files_found lists exact files used

    IMPORTANT: This does NOT “pretend” there are no decisions if reads fail.
               Any read failure raises immediately.
    """
    artifacts_root = artifacts_root.expanduser().resolve()
    dirs = tuple(Path(d).expanduser().resolve() for d in searched_dirs)

    files = find_decision_files(artifacts_root, dirs)

    if verbose:
        print("[Consolidation] run_id:", run_id)
        print("[Consolidation] artifacts_root:", str(artifacts_root))
        print("[Consolidation] searched_dirs:")
        for d in dirs:
            print("  -", str(d))
        print("[Consolidation] decision_files_found:", len(files))
        for f in files[:50]:
            print("  -", str(f))
        if len(files) > 50:
            print(f"  ... ({len(files)-50} more)")

    if not files:
        return ConsolidationResult(
            run_id=run_id,
            artifacts_root=artifacts_root,
            searched_dirs=dirs,
            decision_files_found=tuple(),
            consolidated_path=None,
            rows_written=0,
            unique_source_files=0,
        )

    frames: List[pd.DataFrame] = []
    for p in files:
        df = _read_decisions_file(p)

        # optional schema check
        if require_columns:
            missing = [c for c in require_columns if c not in df.columns]
            if missing:
                raise RuntimeError(
                    f"[Consolidation] File missing required columns {missing}: {p}"
                )

        # annotate provenance
        df = df.copy()
        df["_source_file"] = str(p)

        frames.append(df)

    merged = pd.concat(frames, axis=0, ignore_index=True)

    # best-effort dedupe (only if those columns exist)
    merged = _dedupe_keep_last(merged, dedupe_key_cols)

    out_path = (artifacts_root / output_relpath).resolve()
    _ensure_parent_dir(out_path)

    merged.to_parquet(out_path, index=False)

    rows_written = int(len(merged))
    unique_sources = int(merged["_source_file"].nunique()) if "_source_file" in merged.columns else 0

    if verbose:
        print("[Consolidation] consolidated_path:", str(out_path))
        print("[Consolidation] rows_written:", rows_written)
        print("[Consolidation] unique_source_files:", unique_sources)

    return ConsolidationResult(
        run_id=run_id,
        artifacts_root=artifacts_root,
        searched_dirs=dirs,
        decision_files_found=tuple(files),
        consolidated_path=out_path,
        rows_written=rows_written,
        unique_source_files=unique_sources,
    )
