from __future__ import annotations

import json
from pathlib import Path
import pytest

from feature_engineering.utils.artifacts_root import (
    write_run_manifest,
    resolve_run_dir_from_artifacts_root,
    RUN_MANIFEST_NAME,
)

def test_manifest_required_and_single_truth(tmp_path: Path):
    run_a = tmp_path / "artifacts" / "a2_AAAAA"
    run_b = tmp_path / "artifacts" / "a2_BBBBB"

    # Create both manifests
    write_run_manifest(run_dir=run_a, run_id="AAAAA", artifacts_root=run_a, config_hash="cfgA", clock_hash="clkA")
    write_run_manifest(run_dir=run_b, run_id="BBBBB", artifacts_root=run_b, config_hash="cfgB", clock_hash="clkB")

    # Pass: each run resolves to itself
    rd_a, m_a = resolve_run_dir_from_artifacts_root(run_a)
    assert rd_a == run_a
    assert (run_a / RUN_MANIFEST_NAME).exists()
    assert m_a.run_id == "AAAAA"

    rd_b, m_b = resolve_run_dir_from_artifacts_root(run_b)
    assert rd_b == run_b
    assert (run_b / RUN_MANIFEST_NAME).exists()
    assert m_b.run_id == "BBBBB"

def test_missing_manifest_hard_fails(tmp_path: Path):
    bare = tmp_path / "artifacts" / "a2_NOPE"
    bare.mkdir(parents=True, exist_ok=True)

    with pytest.raises(AssertionError) as e:
        resolve_run_dir_from_artifacts_root(bare)

    assert "Missing RUN_MANIFEST.json" in str(e.value) or "No RUN_MANIFEST.json" in str(e.value)
