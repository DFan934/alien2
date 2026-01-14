from pathlib import Path
import json
import pytest

from feature_engineering.utils.artifacts_root import (
    write_run_manifest,
    assert_manifest_matches_run_dir,
    resolve_run_dir_from_artifacts_root,
    RUN_MANIFEST_NAME,
)

def _seed_manifest(run_dir: Path, run_id: str):
    run_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest(
        run_dir=run_dir,
        run_id=run_id,
        artifacts_root=run_dir,
        config_hash="cfg123",
        clock_hash="clk123",
        cwd=str(Path.cwd()),
    )

def test_task4_cross_run_artifacts_root_mismatch_hard_fails(tmp_path: Path):
    # Arrange: two distinct run dirs
    run_a = tmp_path / "artifacts" / "a2_RUNA"
    run_b = tmp_path / "artifacts" / "a2_RUNB"
    _seed_manifest(run_a, "RUNA")
    _seed_manifest(run_b, "RUNB")

    # Act + Assert:
    # If someone tries to "use" run_a artifacts while expecting run_b, it must hard-fail
    with pytest.raises(AssertionError) as e:
        assert_manifest_matches_run_dir(run_a, expected_run_dir=run_b)

    s = str(e.value)
    assert "RUN_MANIFEST mismatch" in s

def test_task4_resolve_requires_manifest(tmp_path: Path):
    # Arrange: directory without manifest
    naked = tmp_path / "artifacts" / "a2_NAKED"
    naked.mkdir(parents=True, exist_ok=True)

    # Act + Assert: must refuse bare dirs
    with pytest.raises(AssertionError) as e:
        resolve_run_dir_from_artifacts_root(naked)

    assert RUN_MANIFEST_NAME in str(e.value)
