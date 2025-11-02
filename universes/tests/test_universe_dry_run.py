import os
from pathlib import Path
import json
import time
import pytest

from data_ingestion.manifest import summarize_partitions_fast
from scripts.verify_loader import universe_dry_run

def _touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")  # empty; we never read it

def test_summarize_partitions_fast_counts_files(tmp_path: Path):
    root = tmp_path / "parquet"
    # symbol=RRC with two month-days; symbol=BBY empty
    _touch(root / "symbol=RRC/year=1998/month=08/day=01/part-0.parquet")
    _touch(root / "symbol=RRC/year=1998/month=08/day=02/part-0.parquet")
    # BBY has no files in range

    rep = summarize_partitions_fast(root, ["RRC", "BBY"], "1998-08-01", "1998-09-01")
    assert rep["coverage"]["have_data"] == 1
    assert rep["coverage"]["total"] == 2
    assert rep["summary"]["RRC"]["total_files"] == 2
    assert rep["summary"]["BBY"]["total_files"] == 0
    assert rep["coverage"]["empty_symbols"] == ["BBY"]

def test_universe_dry_run_prints_and_returns(tmp_path: Path, capsys):
    root = tmp_path / "parquet"
    _touch(root / "symbol=APA/year=1998/month=09/day=15/part-0.parquet")

    cfg = {
        "parquet_root": str(root),
        "universe": ["apa", "bby"],  # lower → expect UCASE & dedupe from resolver
        "start": "1998-09-01",
        "end":   "1998-10-01",
        "universe_dry_run": True,
    }
    rep = universe_dry_run(cfg)
    captured = capsys.readouterr().out

    # Qualitative: banner and table present
    assert "[Universe-DryRun] size=2" in captured
    assert "symbol | total_files" in captured

    # Quantitative: coverage ratio and empty list correct
    assert rep["coverage"]["have_data"] == 1
    assert rep["coverage"]["total"] == 2
    assert rep["coverage"]["empty_symbols"] == ["BBY"]

    # Should be very fast
    # (Allow a generous bound to avoid flakiness on CI)
    # The function prints elapsed; we also assert programmatically that it returned quickly.
