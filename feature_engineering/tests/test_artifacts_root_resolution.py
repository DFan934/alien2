# tests/test_phase3_artifacts_root_resolution.py
from __future__ import annotations

from pathlib import Path
import os

import pytest

# Adjust import path to wherever you defined resolve_artifacts_root
# Example: from utils.paths import resolve_artifacts_root
from feature_engineering.utils.artifacts_root import resolve_artifacts_root


def _find_repo_root_from_file() -> Path:
    """
    Find the real repo root by walking upward until we hit a project marker.
    Markers: pyproject.toml OR .git OR requirements.txt.
    This avoids wrong assumptions when tests live under a subpackage folder.
    """
    cur = Path(__file__).resolve().parent
    for _ in range(12):
        if (cur / "pyproject.toml").exists() or (cur / ".git").exists() or (cur / "requirements.txt").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    # Fallback: the directory that contains "feature_engineering" as a child is usually the repo root
    cur = Path(__file__).resolve().parent
    for _ in range(12):
        if (cur / "feature_engineering").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return Path(__file__).resolve().parents[2]


def test_resolve_artifacts_root_is_repo_relative_not_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Regression test: running from scripts/ must NOT cause artifacts to land in scripts/artifacts/.
    This catches: resolve_artifacts_root using CWD implicitly.
    """
    repo = _find_repo_root_from_file()
    scripts_dir = repo / "scripts"

    # If scripts/ doesn't exist in your repo, create a temp stand-in
    # but prefer the real one if present.
    if not scripts_dir.exists():
        scripts_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.chdir(scripts_dir)  # simulate `python scripts/run_backtest.py`

    cfg = {"artifacts_root": "artifacts/a2_TEST_PHASE3"}
    root = resolve_artifacts_root(cfg, create=False)

    expected = (repo / "artifacts" / "a2_TEST_PHASE3").resolve()
    assert root.resolve() == expected, f"Expected {expected}, got {root} (cwd={Path.cwd()})"

    # Extra guard: ensure we did NOT accidentally point inside scripts/
    assert "scripts" not in str(root.resolve()).lower(), f"Artifacts root incorrectly under scripts/: {root}"


def test_phase3_no_dual_root_in_log_text() -> None:
    """
    Pure string-level guard: a single run log should not contain both artifacts\\a2_ and scripts\\artifacts\\a2_.
    Replace LOG_TEXT by reading your latest backtest output in CI if desired.
    """
    # If you want: load from an env var or a known output file in artifacts.
    LOG_TEXT = ""
    if not LOG_TEXT:
        return  # opt-in only; see note below

    has_artifacts = ("\\artifacts\\a2_" in LOG_TEXT) or ("/artifacts/a2_" in LOG_TEXT)
    has_scripts_artifacts = ("\\scripts\\artifacts\\a2_" in LOG_TEXT) or ("/scripts/artifacts/a2_" in LOG_TEXT)

    assert not (has_artifacts and has_scripts_artifacts), "Dual artifact roots detected in same run log."
