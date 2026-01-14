# prediction_engine/run_context.py
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional
import json
from feature_engineering.utils.artifacts_root import write_run_manifest


class ArtifactRootMismatchError(RuntimeError):
    def __init__(self, *, expected_root: Path, got_path: Path, rel: str):
        super().__init__(
            f"[ArtifactRootMismatchError] resolved path is outside artifacts_root\n"
            f"  expected_root: {expected_root}\n"
            f"  got_path:      {got_path}\n"
            f"  rel:           {rel}\n"
        )
        self.expected_root = expected_root
        self.got_path = got_path
        self.rel = rel


def _find_repo_root(start: Path) -> Path:
    """
    Repo root = a parent containing scripts/run_backtest.py (preferred) or .git.
    Matches the logic already used in tests like test_signal_density_sanity.py.
    """
    cur = start.resolve()
    for p in [cur, *cur.parents]:
        if (p / "scripts" / "run_backtest.py").exists():
            return p
        if (p / ".git").exists():
            return p
    return cur.parents[0]


def _canonicalize_artifacts_root(root: Path) -> Path:
    """
    Canonical policy for this repo:

    - artifacts_root is always under <repo_root>/artifacts (never <repo_root>/scripts/artifacts)
    - if an input root points at scripts/artifacts, rewrite it to artifacts

    This prevents the exact split observed in logs:
      ...\\scripts\\artifacts\\a2_... vs ...\\artifacts\\a2_...
    """
    root = root.expanduser().resolve()
    parts = list(root.parts)

    # Detect ".../<repo>/scripts/artifacts/..." and rewrite to ".../<repo>/artifacts/..."
    # This is intentionally strict: we do not allow scripts/artifacts as a “valid” root.
    try:
        i = parts.index("scripts")
        if i + 1 < len(parts) and parts[i + 1] == "artifacts":
            # Replace the "scripts/artifacts" segment with just "artifacts"
            new_parts = parts[:i] + ["artifacts"] + parts[i + 2 :]
            return Path(*new_parts).resolve()
    except ValueError:
        pass

    return root


'''@dataclass(frozen=True)
class RunContext:
    run_id: str
    artifacts_root: Path
    universe_hash: Optional[str] = None
    window: Optional[str] = None  # (optional) human-readable window, not a source of truth
'''

def _stable_config_hash(cfg: Mapping[str, Any]) -> str:
    """
    Minimal-stable hash: only uses fields that are deterministic and meaningful.
    Avoids hashing object reprs (which can include memory addresses).
    """
    u = cfg.get("universe")
    if hasattr(u, "symbols"):
        universe = list(u.symbols)
    elif isinstance(u, (list, tuple)):
        universe = [str(x) for x in u]
    else:
        universe = [str(u)]

    payload = {
        "start": str(cfg.get("start", "")),
        "end": str(cfg.get("end", "")),
        "universe": universe,
        # include a few key knobs if present (safe + deterministic)
        "execution_rules": cfg.get("execution_rules", None),
        "scanner": cfg.get("scanner", None),
        "fe": cfg.get("fe", None),
        "strategy": cfg.get("strategy", None),
        "costs": cfg.get("costs", None),
        "gates": cfg.get("gates", None),
        "unified_clock_policy": cfg.get("unified_clock_policy", None),
        "min_clock_symbols": cfg.get("min_clock_symbols", None),
    }
    s = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(s).hexdigest()


@dataclass(frozen=True)
class RunContext:
    run_id: str
    artifacts_root: Path      # base root (absolute)
    run_dir: Path             # per-run directory under artifacts_root
    universe_hash: Optional[str] = None
    window: Optional[str] = None  # (optional) human-readable window


    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        cfg: Mapping[str, Any],
        universe_hash: Optional[str] = None,
        window: Optional[str] = None,
    ) -> "RunContext":
        """
        Single constructor for the run’s artifacts root.

        Rules:
        - If cfg['artifacts_root'] is an explicit run dir (a2_...), accept it (but canonicalize scripts/artifacts → artifacts).
        - Else derive <repo_root>/artifacts/a2_<timestamp> (or run_id driven), but ONLY ONCE here.
        - Immediately persist run_context.json so downstream phases can sanity-check.
        """
        # Prefer: caller already set cfg["artifacts_root"] to the run directory
        '''raw = cfg.get("artifacts_root")

        if raw:
            root = Path(str(raw))
            root = _canonicalize_artifacts_root(root)
        else:
            # Derive default root: <repo_root>/artifacts/<run_id or a2_*>
            repo = _find_repo_root(Path(__file__))
            base = repo / "artifacts"
            # If run_id is already the folder name, use it; else keep a2_ naming external.
            root = base / str(run_id)
            root = _canonicalize_artifacts_root(root)

        root.mkdir(parents=True, exist_ok=True)

        ctx = cls(run_id=str(run_id), artifacts_root=root, universe_hash=universe_hash, window=window)

        # Persist run context for later tooling + debugging + “is this the right run dir?”
        (root / "run_context.json").write_text(
            json.dumps(
                {
                    "run_id": ctx.run_id,
                    "artifacts_root": str(ctx.artifacts_root),
                    "universe_hash": ctx.universe_hash,
                    "window": ctx.window,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        print("[RunContext] artifacts_root=" + str(ctx.artifacts_root))
        return ctx'''


        raw = cfg.get("artifacts_root") or cfg.get("artifacts_dir")
        if raw:
            base = Path(str(raw))
            base = _canonicalize_artifacts_root(base)
        else:
            repo = _find_repo_root(Path(__file__))
            base = (repo / "artifacts").resolve()

        # Hard invariant: base root must be absolute
        base = base.expanduser().resolve()
        if not base.is_absolute():
            raise RuntimeError(f"[RunContext] artifacts_root must be absolute, got: {base}")

        base.mkdir(parents=True, exist_ok=True)

        # Compute run_dir: always a unique per-run folder
        # Policy: run folder name is "a2_<run_id>" unless run_id already starts with "a2_"
        run_name = str(run_id)
        if not run_name.startswith("a2_"):
            run_name = f"a2_{run_name}"
        run_dir = _canonicalize_artifacts_root(base / run_name)

        # Collision must explode loudly
        run_dir.mkdir(parents=True, exist_ok=False)

        # ---------------- Task 4: run-level manifest (single truth) ----------------
        # Manifest must live in the run_dir, and artifacts_root in the manifest must
        # point to the same run_dir to prevent cross-run contamination.

        config_hash = str(cfg.get("config_hash") or cfg.get("_config_hash") or "")
        clock_hash = str(cfg.get("unified_clock_hash") or cfg.get("_unified_clock_hash") or "")

        cfg_config_hash = str(cfg.get("config_hash") or "").strip()
        if not cfg_config_hash:
            cfg_config_hash = _stable_config_hash(cfg)

        # Ensure downstream stages can read it (Task 4 requires config_hash always known at run start)
        if isinstance(cfg, dict):
            cfg["_config_hash"] = cfg_config_hash
            cfg["config_hash"] = cfg_config_hash  # optional, but makes intent unmissable



        manifest_path = write_run_manifest(
            run_dir=run_dir,
            run_id=str(run_id),
            artifacts_root=run_dir,   # single truth: for this run, root == run_dir
            #config_hash=config_hash,
            config_hash=cfg_config_hash,
            clock_hash=clock_hash,
            cwd=str(Path.cwd()),
        )

        print(
            f"[RunManifest] path={str(manifest_path.resolve())} "
            f"config_hash={cfg_config_hash} clock_hash={clock_hash}"
        )

        print(f"[RunManifest] path={manifest_path} config_hash={config_hash} clock_hash={clock_hash}")
        # ---------------- end Task 4 ------------------------------------------------



        ctx = cls(
            run_id=str(run_id),
            artifacts_root=base,
            run_dir=run_dir,
            universe_hash=universe_hash,
            window=window,
        )

        # Persist run context (in run_dir, not base)
        (run_dir / "run_context.json").write_text(
            json.dumps(
                {
                    "run_id": ctx.run_id,
                    "artifacts_root": str(ctx.artifacts_root),
                    "run_dir": str(ctx.run_dir),
                    "universe_hash": ctx.universe_hash,
                    "window": ctx.window,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        # Required: tiny file containing the absolute root + run_dir
        (run_dir / "_ARTIFACTS_ROOT.txt").write_text(
            f"artifacts_root={ctx.artifacts_root}\nrun_dir={ctx.run_dir}\n",
            encoding="utf-8",
        )

        print("[RunContext] artifacts_root=" + str(ctx.artifacts_root))
        print("[RunContext] run_dir=" + str(ctx.run_dir))
        return ctx


    def resolve(self, rel: str) -> Path:
        """
        Resolve a relative path under artifacts_root with a HARD invariant:
        resolved_path MUST be inside artifacts_root.

        This is the guardrail that makes scripts\\artifacts divergence impossible.
        """
        '''rel = str(rel).lstrip("/\\")
        p = (self.artifacts_root / rel).resolve()

        # Invariant: p is a child of artifacts_root
        root = self.artifacts_root.resolve()
        try:
            p.relative_to(root)
        except Exception:
            raise ArtifactRootMismatchError(expected_root=root, got_path=p, rel=rel)

        return p'''

        rel = str(rel).lstrip("/\\")
        p = (self.run_dir / rel).resolve()

        root = self.run_dir.resolve()
        try:
            p.relative_to(root)
        except Exception:
            raise ArtifactRootMismatchError(expected_root=root, got_path=p, rel=rel)

        return p

