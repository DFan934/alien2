from pathlib import Path
import tempfile
import pytest

from feature_engineering.utils.artifacts_root import (
    write_run_manifest,
    read_run_manifest,
    resolve_run_dir_from_artifacts_root,
    assert_manifest_matches_run_dir,
    update_run_manifest_fields,
    assert_manifest_has_hashes,
)

def test_task4_refuse_bare_dir_without_manifest():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "no_manifest_dir"
        d.mkdir(parents=True, exist_ok=True)
        with pytest.raises(AssertionError, match="Missing RUN_MANIFEST.json"):
            resolve_run_dir_from_artifacts_root(d)

def test_task4_hashes_present_and_crossrun_mismatch_hardfails():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)

        run_a = base / "artifacts" / "a2_AAAA"
        run_b = base / "artifacts" / "a2_BBBB"
        run_a.mkdir(parents=True, exist_ok=True)
        run_b.mkdir(parents=True, exist_ok=True)

        # Write initial manifests (clock may be empty at creation time)
        write_run_manifest(
            run_dir=run_a,
            run_id="AAAA",
            artifacts_root=run_a,
            config_hash="cfgA",
            clock_hash="",
            cwd=str(base),
        )
        write_run_manifest(
            run_dir=run_b,
            run_id="BBBB",
            artifacts_root=run_b,
            config_hash="cfgB",
            clock_hash="",
            cwd=str(base),
        )

        # Update A to simulate Task3 producing the clock hash later
        update_run_manifest_fields(run_a, clock_hash="clockA")

        # Must now satisfy Task4 hash requirements
        m = assert_manifest_has_hashes(run_a)
        assert m.config_hash == "cfgA"
        assert m.clock_hash == "clockA"

        # Cross-run misuse must hard-fail
        with pytest.raises(AssertionError, match="RUN_MANIFEST mismatch"):
            assert_manifest_matches_run_dir(run_a, expected_run_dir=run_b)

        # Sanity: B still missing clock hash, should fail hash assertion
        with pytest.raises(AssertionError, match="missing clock_hash"):
            assert_manifest_has_hashes(run_b)
