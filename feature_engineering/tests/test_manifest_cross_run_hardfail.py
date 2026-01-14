from __future__ import annotations

from pathlib import Path
import pytest

from feature_engineering.utils.artifacts_root import (
    write_run_manifest,
    require_run_manifest,
    resolve_run_dir,
)


def test_task4_cross_run_manifest_hardfails(tmp_path: Path):
    base = tmp_path / "artifacts"
    base.mkdir()

    run_a = base / "a2_A"
    run_b = base / "a2_B"
    run_a.mkdir()
    run_b.mkdir()

    write_run_manifest(
        run_dir=run_a, run_id="A", artifacts_root=base,
        config_hash="cfgAAA", clock_hash="clkAAA",
    )
    write_run_manifest(
        run_dir=run_b, run_id="B", artifacts_root=base,
        config_hash="cfgBBB", clock_hash="clkBBB",
    )

    # If cfg says run_id=B but run_dir points to A, resolve_run_dir must fail
    cfg = {"run_id": "B", "run_dir": str(run_a)}

    with pytest.raises(RuntimeError, match=r"RUN_MANIFEST mismatch: run_id"):
        resolve_run_dir(cfg, require_manifest=True)

    # Direct require also must fail
    with pytest.raises(RuntimeError, match=r"RUN_MANIFEST mismatch: run_id"):
        require_run_manifest(run_dir=run_a, expected_run_id="B")
