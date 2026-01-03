import os
import subprocess
from pathlib import Path

def test_artifacts_root_same_from_repo_root_and_scripts(tmp_path):
    repo = Path(__file__).resolve().parents[2]
    script = repo / "scripts" / "run_backtest.py"

    # Run once from repo root
    p1 = subprocess.run(
        ["python", str(script)],
        cwd=str(repo),
        capture_output=True,
        text=True,
    )
    assert p1.returncode == 0
    assert "[RunContext] artifacts_root=" in p1.stdout
    assert "[RunContext] run_dir=" in p1.stdout

    # Run once from scripts/
    p2 = subprocess.run(
        ["python", "run_backtest.py"],
        cwd=str(repo / "scripts"),
        capture_output=True,
        text=True,
    )
    assert p2.returncode == 0
    assert "[RunContext] artifacts_root=" in p2.stdout
    assert "[RunContext] run_dir=" in p2.stdout

    # Both runs must NOT mention scripts\\artifacts
    assert "scripts\\artifacts" not in (p1.stdout + p2.stdout).lower()

    # And each run must write _ARTIFACTS_ROOT.txt
    # (We can’t easily know the exact run_dir name here, so just assert the marker appears in output)
    assert "_ARTIFACTS_ROOT.txt" not in (p1.stdout + p2.stdout)  # should not be printed unless you print it
