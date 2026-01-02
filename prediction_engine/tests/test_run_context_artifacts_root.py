# prediction_engine/tests/test_run_context_artifacts_root.py
from pathlib import Path
import tempfile

import pytest

from prediction_engine.run_context import RunContext, ArtifactRootMismatchError


def test_resolve_rejects_escape():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td).resolve()

        # Construct a ctx with an explicit artifacts_root
        ctx = RunContext.create(run_id="a2_TEST", cfg={"artifacts_root": str(root / "a2_TEST")})

        # Attempt to escape via .. should hard fail
        with pytest.raises(ArtifactRootMismatchError):
            ctx.resolve("../outside.txt")


def test_canonicalize_rewrites_scripts_artifacts():
    with tempfile.TemporaryDirectory() as td:
        repo = Path(td).resolve()
        (repo / "scripts").mkdir(parents=True, exist_ok=True)
        (repo / "artifacts").mkdir(parents=True, exist_ok=True)

        # Simulate user passing scripts/artifacts
        bad = repo / "scripts" / "artifacts" / "a2_TEST"
        ctx = RunContext.create(run_id="a2_TEST", cfg={"artifacts_root": str(bad)})

        # Must rewrite to repo/artifacts/a2_TEST
        assert "scripts" not in ctx.artifacts_root.parts
        assert ctx.artifacts_root.parts[-2:] == ("artifacts", "a2_TEST")


def test_run_context_written():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td).resolve()
        ctx = RunContext.create(run_id="a2_TEST", cfg={"artifacts_root": str(base / "artifacts" / "a2_TEST")})
        assert (ctx.artifacts_root / "run_context.json").exists()
