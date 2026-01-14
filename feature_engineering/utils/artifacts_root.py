from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
from typing import Any, Mapping
import json
import datetime as dt
from dataclasses import dataclass


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


# -------------------- Task 4: RUN_MANIFEST single truth --------------------

RUN_MANIFEST_NAME = "RUN_MANIFEST.json"

@dataclass(frozen=True)
class RunManifest:
    run_id: str
    run_dir: str
    artifacts_root: str
    created_at: str
    cwd: str
    config_hash: str
    clock_hash: str

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "RunManifest":
        # tolerate older manifests missing new fields by defaulting to empty string
        return RunManifest(
            run_id=str(d.get("run_id", "")),
            run_dir=str(d.get("run_dir", "")),
            artifacts_root=str(d.get("artifacts_root", "")),
            created_at=str(d.get("created_at", "")),
            cwd=str(d.get("cwd", "")),
            config_hash=str(d.get("config_hash", "")),
            clock_hash=str(d.get("clock_hash", "")),
        )

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "artifacts_root": self.artifacts_root,
            "created_at": self.created_at,
            "cwd": self.cwd,
            "config_hash": self.config_hash,
            "clock_hash": self.clock_hash,
        }


def _norm(p: Path | str) -> str:
    return str(Path(p).resolve()).lower().replace("/", "\\")


def find_run_manifest(start: Path) -> Path | None:
    """
    Search upward from `start` for RUN_MANIFEST.json.
    Returns the first found manifest path, else None.
    """
    cur = Path(start).resolve()
    for _ in range(12):
        cand = cur / RUN_MANIFEST_NAME
        if cand.exists():
            return cand
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def read_run_manifest(run_dir: Path) -> RunManifest:
    mpath = Path(run_dir) / RUN_MANIFEST_NAME
    if not mpath.exists():
        raise AssertionError(f"[Task4] Missing {RUN_MANIFEST_NAME} in run_dir={run_dir}")
    try:
        data = json.loads(mpath.read_text(encoding="utf-8"))
    except Exception as e:
        raise AssertionError(f"[Task4] Failed reading {mpath}: {e}")
    return RunManifest.from_dict(data)


def write_run_manifest(
    *,
    run_dir: Path,
    run_id: str,
    artifacts_root: Path,
    config_hash: str,
    clock_hash: str,
    cwd: str | None = None,
) -> Path:
    run_dir = Path(run_dir).resolve()
    artifacts_root = Path(artifacts_root).resolve()
    mpath = run_dir / RUN_MANIFEST_NAME
    payload = RunManifest(
        run_id=str(run_id),
        run_dir=str(run_dir),
        artifacts_root=str(artifacts_root),
        created_at=dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        cwd=str(cwd or Path.cwd()),
        config_hash=str(config_hash),
        clock_hash=str(clock_hash),
    ).to_dict()
    run_dir.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return mpath


def update_run_manifest_fields(
    run_dir: Path | str,
    *,
    config_hash: str | None = None,
    clock_hash: str | None = None,
) -> Path:
    """
    Update an existing RUN_MANIFEST.json in-place.
    Used when fields become known later in the run (e.g., clock_hash after Task 3).
    """
    run_dir = Path(run_dir).resolve()
    mpath = run_dir / RUN_MANIFEST_NAME
    if not mpath.exists():
        raise AssertionError(f"[Task4] Missing {RUN_MANIFEST_NAME} in run_dir={run_dir}")

    try:
        data = json.loads(mpath.read_text(encoding="utf-8"))
    except Exception as e:
        raise AssertionError(f"[Task4] Failed reading {mpath}: {e}")

    if config_hash is not None:
        data["config_hash"] = str(config_hash)
    if clock_hash is not None:
        data["clock_hash"] = str(clock_hash)

    mpath.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return mpath


def assert_manifest_has_hashes(run_dir: Path | str) -> RunManifest:
    """
    Hard requirement for Task 4 completion:
      - config_hash must be non-empty
      - clock_hash must be non-empty
    """
    run_dir = Path(run_dir).resolve()
    m = assert_manifest_matches_run_dir(run_dir)

    if not str(m.config_hash).strip():
        raise AssertionError(f"[Task4] RUN_MANIFEST missing config_hash in run_dir={run_dir}")
    if not str(m.clock_hash).strip():
        raise AssertionError(f"[Task4] RUN_MANIFEST missing clock_hash in run_dir={run_dir}")

    return m



def assert_manifest_matches_run_dir(run_dir: Path, *, expected_run_dir: Path | None = None) -> RunManifest:
    """
    Hard-fail if:
      - manifest missing
      - manifest.run_dir != run_dir (or expected_run_dir if provided)
    """
    run_dir = Path(run_dir).resolve()
    m = read_run_manifest(run_dir)

    want = Path(expected_run_dir).resolve() if expected_run_dir is not None else run_dir

    if _norm(m.run_dir) != _norm(want):
        raise AssertionError(
            f"[Task4] RUN_MANIFEST mismatch: manifest.run_dir={m.run_dir} != expected={want}"
        )

    # extra: enforce artifacts_root points to the same run_dir (single truth)
    if _norm(m.artifacts_root) != _norm(run_dir):
        raise AssertionError(
            f"[Task4] RUN_MANIFEST mismatch: manifest.artifacts_root={m.artifacts_root} != run_dir={run_dir}"
        )

    return m


def require_manifest_for_path(p: Path, *, expected_run_dir: Path | None = None) -> RunManifest:
    """
    Given any path inside a run, find the run manifest and assert it's valid.
    """
    p = Path(p).resolve()
    mpath = find_run_manifest(p)
    if mpath is None:
        raise AssertionError(f"[Task4] No {RUN_MANIFEST_NAME} found above path={p}")
    run_dir = mpath.parent

    return assert_manifest_matches_run_dir(run_dir, expected_run_dir=expected_run_dir)

    #return assert_manifest_matches_run_dir(run_dir)


def resolve_run_dir_from_artifacts_root(artifacts_root: Path | str) -> tuple[Path, RunManifest]:
    """
    The single source of truth for 'what run are we in?'
    Must point at a directory that contains RUN_MANIFEST.json.
    """
    run_dir = Path(artifacts_root).resolve()
    m = assert_manifest_matches_run_dir(run_dir)
    return run_dir, m

# ------------------ end Task 4 --------------------------------------------
