# universes/tests/test_universe_plumbing_minimal.py
import io
from pathlib import Path
import pytest

from universes.providers import resolve_universe, StaticUniverse

# We’ll reuse your dry-run utility for coverage summary
from scripts.verify_loader import universe_dry_run
# after the imports (after "from scripts.verify_loader import universe_dry_run")
import pytest
pytestmark = pytest.mark.phase2_minimal

# ---------------------------
# helpers
# ---------------------------
def _touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"PAR1")  # no read needed; we just count files

# ---------------------------
# 1) test_resolve_universe_static
# ---------------------------
def test_resolve_universe_static():
    cfg = {"universe": StaticUniverse(["rrc", "RRC", "bby", "BbY", "apa"])}
    out = resolve_universe(cfg)
    # Quantitative: de-duped + UPPER
    assert out == ["RRC", "BBY", "APA"]

# ---------------------------
# 2) test_resolve_universe_file
#    (covers both .txt and .csv)
# ---------------------------
@pytest.mark.parametrize("ext, payload", [
    ("txt", "rrc\nbby\napa\n"),
    ("csv", "symbol\nrrc\nBBY\nApa\n"),
])
def test_resolve_universe_file(tmp_path: Path, ext: str, payload: str):
    upath = tmp_path / f"u.{ext}"
    upath.write_text(payload, encoding="utf-8")
    cfg = {"universe": str(upath)}
    out = resolve_universe(cfg)
    assert out == ["RRC", "BBY", "APA"]
    # Qualitative: reads cleanly; Quantitative: correct count
    assert len(out) == 3

# ---------------------------
# 3) test_universe_dry_run_summary
#    (2 symbols; 1 has data → 50% coverage)
# ---------------------------
def test_universe_dry_run_summary(tmp_path: Path, capsys):
    root = tmp_path / "parquet"
    # APA has 2 files in window; BBY has none
    _touch(root / "symbol=APA/year=1999/month=01/day=01/part-0.parquet")
    _touch(root / "symbol=APA/year=1999/month=01/day=15/part-0.parquet")

    cfg = {
        "parquet_root": str(root),
        "universe": ["apa", "bby"],  # lower → expect UCASE from resolver
        "start": "1999-01-01",
        "end":   "1999-02-01",
        "universe_dry_run": True,
    }
    rep = universe_dry_run(cfg)
    out = capsys.readouterr().out

    # Qualitative: compact table & banner
    assert "[Universe-DryRun] size=2" in out
    assert "symbol | total_files" in out

    # Quantitative: coverage metrics
    assert rep["coverage"]["have_data"] == 1
    assert rep["coverage"]["total"] == 2
    assert rep["coverage"]["empty_symbols"] == ["BBY"]
    # Should be very fast (CI/leeway friendly): rely on function design; no parquet reads

# ---------------------------
# 4) test_max_size_guard
#    (exceed universe_max_size → fail early)
# ---------------------------
def test_max_size_guard(tmp_path: Path):
    # Dry-run path is simplest place to assert guard firing fast
    cfg = {
        "parquet_root": str(tmp_path / "parquet"),
        "universe": [f"S{i}" for i in range(12)],
        "universe_max_size": 10,
        "start": "1999-01-01",
        "end":   "1999-02-01",
    }
    with pytest.raises(RuntimeError) as ei:
        universe_dry_run(cfg)
    msg = str(ei.value)
    assert "Universe too large" in msg
    assert "> 10" in msg or "10" in msg  # useful diff on failure
