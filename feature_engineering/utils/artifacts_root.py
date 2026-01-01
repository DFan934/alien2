from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def _find_repo_root(anchor: Path) -> Path:
    """
    Walk upward from 'anchor' and return the first directory that looks like a repo root.
    Heuristics: pyproject.toml OR .git OR requirements.txt.
    """
    cur = anchor.resolve()
    for _ in range(10):
        if (cur / "pyproject.toml").exists() or (cur / ".git").exists() or (cur / "requirements.txt").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    # Fallback: best-effort (should still be stable for typical layouts)
    return anchor.resolve()


def resolve_artifacts_root(cfg: Mapping[str, Any], run_id: str | None = None, *, create: bool = True) -> Path:
    """
    Canonical artifact root resolver.

    Rules:
      - Use cfg["artifacts_root"] as single truth (cfg["artifacts_dir"] allowed only as legacy alias).
      - If relative, resolve relative to *repo root* (NOT CWD).
      - Never prepend "scripts/" automatically.
      - Never fall back to secondary roots.
    """
    raw = cfg.get("artifacts_root") or cfg.get("artifacts_dir") or "artifacts/a2"
    p = Path(str(raw))

    # If someone provides "artifacts/a2" with a run_id, optionally suffix it.
    if run_id is not None and not p.is_absolute():
        # only suffix if it looks like a base folder and not already timestamped
        name = p.name
        if name in {"a2", "artifacts"} and "a2_" not in name:
            p = p / f"a2_{run_id}"

    if not p.is_absolute():
        # anchor on this file's location to find repo root deterministically
        repo_root = _find_repo_root(Path(__file__).resolve().parent)
        p = repo_root / p

    if create:
        p.mkdir(parents=True, exist_ok=True)

    return p
