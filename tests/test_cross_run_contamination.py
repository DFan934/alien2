# feature_engineering/tests/test_task4_cross_run_contamination.py

from __future__ import annotations

from pathlib import Path
import pytest

from feature_engineering.utils.artifacts_root import (
    write_run_manifest,
    require_manifest_for_path,
)


def test_task4_cross_run_read_hard_fails(tmp_path: Path) -> None:
    """
    Task 4 quantitative proof:
    Any attempt to read artifacts from a different run_dir MUST hard-fail.
    """

    run_a = (tmp_path / "artifacts" / "a2_runA").resolve()
    run_b = (tmp_path / "artifacts" / "a2_runB").resolve()
    run_a.mkdir(parents=True, exist_ok=True)
    run_b.mkdir(parents=True, exist_ok=True)

    # Create valid manifests for both runs
    write_run_manifest(
        run_dir=run_a,
        run_id="runA",
        artifacts_root=run_a,
        config_hash="cfgA",
        clock_hash="clkA",
    )
    write_run_manifest(
        run_dir=run_b,
        run_id="runB",
        artifacts_root=run_b,
        config_hash="cfgB",
        clock_hash="clkB",
    )

    # Create a dummy artifact inside run B
    artifact_in_b = run_b / "weights" / "calibration" / "dummy.json"
    artifact_in_b.parent.mkdir(parents=True, exist_ok=True)
    artifact_in_b.write_text("{}", encoding="utf-8")

    # Simulate: current run is A, but a loader is pointed at a path inside run B
    with pytest.raises(AssertionError, match=r"RUN_MANIFEST mismatch"):
        require_manifest_for_path(artifact_in_b, expected_run_dir=run_a)
