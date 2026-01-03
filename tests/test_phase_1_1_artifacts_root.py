from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Tuple


def _find_repo_root(anchor: Path) -> Path:
    """Best-effort repo root detection."""
    cur = anchor.resolve()
    for _ in range(12):
        if (cur / "pyproject.toml").exists() or (cur / ".git").exists() or (cur / "requirements.txt").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    # Fallback: assume scripts/ is under repo root
    return anchor.resolve().parent


ARTIFACTS_LINE_RE = re.compile(r"^\[RunContext\]\s+artifacts_root=(.+)$", re.MULTILINE)


def _run_once(repo_root: Path, *, cwd: Path, cmd: list[str]) -> Tuple[int, str]:
    """Run a subprocess, capture combined stdout/stderr text."""
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )
    return p.returncode, p.stdout


def _normalize_slashes(s: str) -> str:
    return s.replace("/", "\\").lower()


def _extract_artifacts_root(output: str) -> Path:
    m = ARTIFACTS_LINE_RE.search(output)
    if not m:
        raise AssertionError(
            "Did not find a line like:\n"
            "  [RunContext] artifacts_root=...\n\n"
            "Make sure your run prints that exactly once.\n"
        )
    raw = m.group(1).strip().strip('"').strip("'")
    return Path(raw).expanduser()


def _assert_no_scripts_artifacts(output: str, label: str) -> None:
    low = _normalize_slashes(output)
    if "\\scripts\\artifacts" in low:
        raise AssertionError(
            f"[{label}] Output contains forbidden substring '\\\\scripts\\\\artifacts'.\n"
            f"This violates Phase 1.1.\n"
        )


def _assert_single_artifacts_root_line(output: str, label: str) -> None:
    matches = ARTIFACTS_LINE_RE.findall(output)
    if len(matches) != 1:
        raise AssertionError(
            f"[{label}] Expected exactly ONE '[RunContext] artifacts_root=...' line, "
            f"found {len(matches)}.\n"
            f"Matches:\n" + "\n".join(f"  {x}" for x in matches[:10])
        )


def _assert_root_is_canonical(repo_root: Path, run_dir: Path, label: str) -> None:
    if not run_dir.is_absolute():
        raise AssertionError(f"[{label}] artifacts_root is not absolute: {run_dir}")

    # Must be under <repo_root>/artifacts
    canonical_base = (repo_root / "artifacts").resolve()
    run_dir_resolved = run_dir.resolve()

    # A safe "is under" check
    try:
        run_dir_resolved.relative_to(canonical_base)
    except Exception:
        raise AssertionError(
            f"[{label}] artifacts_root is not under canonical base.\n"
            f"  expected base: {canonical_base}\n"
            f"  got run_dir    : {run_dir_resolved}\n"
        )

    # Must not be under <repo_root>/scripts/artifacts
    forbidden_base = (repo_root / "scripts" / "artifacts").resolve()
    try:
        run_dir_resolved.relative_to(forbidden_base)
        raise AssertionError(
            f"[{label}] artifacts_root is under forbidden base scripts/artifacts:\n"
            f"  forbidden base: {forbidden_base}\n"
            f"  got run_dir    : {run_dir_resolved}\n"
        )
    except Exception:
        # good: not under forbidden base
        pass

    # Must have _ARTIFACTS_ROOT.txt in the run directory
    marker = run_dir_resolved / "_ARTIFACTS_ROOT.txt"
    if not marker.exists():
        raise AssertionError(
            f"[{label}] Missing required marker file:\n"
            f"  {marker}\n"
            f"Phase 1.1 requires the run dir to contain _ARTIFACTS_ROOT.txt.\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 1.1 integration test: artifacts_root canonicalization across CWDs."
    )
    parser.add_argument(
        "--",
        dest="passthrough",
        nargs=argparse.REMAINDER,
        help="Arguments to pass through to run_backtest.py (use after '--').",
    )
    args = parser.parse_args()

    # argparse puts the separator itself in passthrough sometimes; clean it.
    passthrough = list(args.passthrough or [])
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    this_file = Path(__file__).resolve()
    repo_root = _find_repo_root(this_file.parent)

    run_py_repo = repo_root / "scripts" / "run_backtest.py"
    run_py_scripts = repo_root / "scripts" / "run_backtest.py"  # same file, different CWD

    if not run_py_repo.exists():
        raise SystemExit(f"Cannot find run_backtest.py at: {run_py_repo}")

    # --- Run A: from repo root ---
    cmd_a = [sys.executable, str(run_py_repo)] + passthrough
    code_a, out_a = _run_once(repo_root, cwd=repo_root, cmd=cmd_a)

    # --- Run B: from scripts/ ---
    cmd_b = [sys.executable, str(run_py_scripts.name)] + passthrough
    code_b, out_b = _run_once(repo_root, cwd=(repo_root / "scripts"), cmd=cmd_b)

    # Basic success on process exit
    if code_a != 0:
        print("=== RUN A FAILED (repo root) ===")
        print(out_a[-4000:])
        raise SystemExit(code_a)

    if code_b != 0:
        print("=== RUN B FAILED (scripts/ cwd) ===")
        print(out_b[-4000:])
        raise SystemExit(code_b)

    # Phase 1.1 checks
    _assert_no_scripts_artifacts(out_a, "RUN A")
    _assert_no_scripts_artifacts(out_b, "RUN B")

    _assert_single_artifacts_root_line(out_a, "RUN A")
    _assert_single_artifacts_root_line(out_b, "RUN B")

    run_dir_a = _extract_artifacts_root(out_a)
    run_dir_b = _extract_artifacts_root(out_b)

    _assert_root_is_canonical(repo_root, run_dir_a, "RUN A")
    _assert_root_is_canonical(repo_root, run_dir_b, "RUN B")

    print("PASS: Phase 1.1 artifacts_root integration test")
    print(f"  repo_root      : {repo_root}")
    print(f"  run_dir (A)     : {run_dir_a.resolve()}")
    print(f"  run_dir (B)     : {run_dir_b.resolve()}")
    print(f"  canonical base  : {(repo_root / 'artifacts').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
