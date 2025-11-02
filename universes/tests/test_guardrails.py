import pytest
from pathlib import Path
import pytest
pytestmark = pytest.mark.phase2_minimal

# We’ll exercise both scripts to cover all guardrails.
from scripts.verify_loader import universe_dry_run
from scripts.run_backtest import run as run_backtest

# Helper: build a tiny parquet hive so summarize_partitions_fast finds something
def _seed_parquet(tmp: Path, sym: str, y=1999, m=1, d=1):
    p = tmp / f"symbol={sym}/year={y}/month={m:02d}/day={d:02d}"
    p.mkdir(parents=True, exist_ok=True)
    (p / "part-0.parquet").write_bytes(b"PAR1")  # header bytes enough for globbing path tests

def test_empty_universe_fails_dry_run(tmp_path):
    cfg = {
        "parquet_root": str(tmp_path / "parquet"),
        "universe": [], "start": "1999-01-01", "end": "1999-02-01",
        "universe_max_size": 10, "strict_universe": True,
    }
    with pytest.raises(RuntimeError) as e:
        universe_dry_run(cfg)
    assert "Universe is empty" in str(e.value)

@pytest.mark.asyncio
async def test_empty_universe_fails_runner(tmp_path):
    cfg = {
        "parquet_root": str(tmp_path / "parquet"),
        "universe": [], "start": "1999-01-01", "end": "1999-02-01",
        "universe_max_size": 10, "artifacts_root": str(tmp_path / "art"),
    }
    with pytest.raises(RuntimeError) as e:
        await run_backtest(cfg)
    assert "Universe is empty" in str(e.value)

def test_max_size_guard_dry_run(tmp_path):
    cfg = {
        "parquet_root": str(tmp_path / "parquet"),
        "universe": ["S"+str(i) for i in range(12)],
        "universe_max_size": 10,
        "start": "1999-01-01", "end": "1999-02-01",
    }
    with pytest.raises(RuntimeError) as e:
        universe_dry_run(cfg)
    assert "Universe too large" in str(e.value)

def test_coverage_guard_dry_run_strict(tmp_path):
    root = tmp_path / "parquet"; root.mkdir()
    # Only APA has a partition; BBY empty → 50% coverage
    _seed_parquet(root, "APA")
    cfg = {
        "parquet_root": str(root),
        "universe": ["APA", "BBY"],
        "start": "1999-01-01", "end": "1999-02-01",
        "strict_universe": True,
        "universe_min_coverage": 0.90,
    }
    with pytest.raises(RuntimeError) as e:
        universe_dry_run(cfg)
    s = str(e.value)
    assert "Coverage" in s and "Empty symbols" in s

@pytest.mark.asyncio
async def test_coverage_guard_runner_strict(tmp_path):
    root = tmp_path / "parquet"; root.mkdir()
    _seed_parquet(root, "APA")
    cfg = {
        "parquet_root": str(root),
        "universe": ["APA", "BBY"],
        "start": "1999-01-01", "end": "1999-02-01",
        "strict_universe": True,
        "universe_min_coverage": 0.90,
        "artifacts_root": str(tmp_path / "art"),
    }
    with pytest.raises(RuntimeError) as e:
        await run_backtest(cfg)
    s = str(e.value)
    assert "Coverage" in s and "Empty symbols" in s
