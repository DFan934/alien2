from __future__ import annotations

from pathlib import Path

from feature_engineering.utils.artifacts_root import resolve_artifacts_root


def test_artifacts_root_resolves_relative_to_repo_not_cwd(tmp_path: Path):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"

    repo.mkdir(parents=True, exist_ok=True)  # <-- ADD THIS
    (repo / "pyproject.toml").write_text("dummy = true")

    scripts.mkdir(parents=True, exist_ok=True)

    cfg = {"artifacts_root": "artifacts/a2_TEST"}

    old = Path.cwd()
    try:
        import os
        os.chdir(scripts)

        out = resolve_artifacts_root(cfg, create=False)

        assert "scripts" not in out.parts, f"Should not resolve under scripts/: got {out}"
        assert "artifacts" in out.parts, f"Expected artifacts path: got {out}"
    finally:
        import os
        os.chdir(old)

