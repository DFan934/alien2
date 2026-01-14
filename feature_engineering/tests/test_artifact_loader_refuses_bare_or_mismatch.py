from __future__ import annotations

from pathlib import Path
import pytest

from feature_engineering.utils.artifacts_root import write_run_manifest
from prediction_engine.prediction_engine.artifacts.loader import resolve_artifact_paths

def test_loader_requires_manifest(tmp_path: Path):
    run_dir = tmp_path / "artifacts" / "a2_RUNX"
    run_dir.mkdir(parents=True, exist_ok=True)

    # No manifest => should hard fail
    with pytest.raises(AssertionError):
        resolve_artifact_paths(artifacts_root=run_dir, symbol="RRC", strategy="pooled")

def test_loader_ok_with_manifest(tmp_path: Path):
    run_dir = tmp_path / "artifacts" / "a2_RUNY"
    write_run_manifest(run_dir=run_dir, run_id="RUNY", artifacts_root=run_dir, config_hash="cfg", clock_hash="clk")

    out = resolve_artifact_paths(artifacts_root=run_dir, symbol="RRC", strategy="pooled")
    assert "core_dir" in out
    assert "calibrator" in out
