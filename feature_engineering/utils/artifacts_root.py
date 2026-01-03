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
    return anchor.resolve()


def _strip_scripts_artifacts(p: Path) -> Path:
    """
    If someone accidentally points to <repo>/scripts/artifacts/...,
    rewrite it to <repo>/artifacts/... (drop the 'scripts' segment).

    This is the single biggest Phase 1.1 footgun.
    """
    try:
        parts = list(p.resolve().parts)
    except Exception:
        parts = list(p.parts)

    lower = [x.lower() for x in parts]
    # find ...\scripts\artifacts\...
    for i in range(len(lower) - 1):
        if lower[i] == "scripts" and lower[i + 1] == "artifacts":
            # keep everything before "scripts", then "artifacts", then everything after "artifacts"
            before = parts[:i]
            after = parts[i + 2 :]
            return Path(*before) / "artifacts" / Path(*after) if after else (Path(*before) / "artifacts")
    return p


def resolve_artifacts_root(cfg: Mapping[str, Any], run_id: str | None = None, *, create: bool = True) -> Path:
    """
    Canonical artifact root resolver.

    Rules:
      - Use cfg["artifacts_root"] as single truth (cfg["artifacts_dir"] allowed only as legacy alias).
      - If relative, resolve relative to *repo root* (NOT CWD).
      - NEVER allow <repo>/scripts/artifacts/...  (rewrite to <repo>/artifacts/...)
      - Never fall back to secondary roots.
    """
    raw = cfg.get("artifacts_root") or cfg.get("artifacts_dir") or "artifacts"
    p = Path(str(raw))

    # If relative, anchor to repo root deterministically.
    if not p.is_absolute():
        repo_root = _find_repo_root(Path(__file__).resolve().parent)
        p = repo_root / p

    # Rewrite accidental scripts/artifacts usage.
    p = _strip_scripts_artifacts(p)

    # Phase 1.1: hard forbid scripts/artifacts, even after rewrite.
    # (Rewrite prevents footguns; this assertion prevents silent provenance drift.)
    norm = str(p).lower().replace("/", "\\")
    if "\\scripts\\artifacts" in norm:
        raise AssertionError(f"[Phase 1.1] Forbidden artifacts root contains scripts\\artifacts: {p}")

    # Optional: suffix run folder if caller asks and path looks like a base folder
    if run_id is not None:
        rid = str(run_id).strip()
        if rid:
            target = f"a2_{rid}".lower()
            name = p.name.lower()

            # If we are already in a2_<run_id>, do nothing.
            if name == target:
                pass
            else:
                # If user passed ".../artifacts" -> ".../artifacts/a2_<run_id>"
                if name == "artifacts":
                    p = p / f"a2_{rid}"
                # If user passed ".../artifacts/a2" -> ".../artifacts/a2_<run_id>"
                elif name == "a2":
                    p = p.parent / f"a2_{rid}"
                # Otherwise: leave p alone (already run-scoped or caller supplied explicit dir)

    # Hard assertion for Phase 1.1
    if not p.is_absolute():
        raise AssertionError(f"[Phase 1.1] artifacts_root must be absolute, got: {p}")

    if create:
        p.mkdir(parents=True, exist_ok=True)

    return p
