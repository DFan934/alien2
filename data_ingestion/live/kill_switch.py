# data_ingestion/live/kill_switch.py
from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_LATCH_NAME = "KILL_SWITCH"


def latch_path(run_dir: Path, *, name: str = DEFAULT_LATCH_NAME) -> Path:
    return Path(run_dir) / name


def is_engaged(run_dir: Path, *, name: str = DEFAULT_LATCH_NAME) -> bool:
    return latch_path(run_dir, name=name).exists()


def engage(run_dir: Path, *, name: str = DEFAULT_LATCH_NAME) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    p = latch_path(run_dir, name=name)
    p.write_text("ENGAGED\n", encoding="utf-8")
    return p


def disengage(run_dir: Path, *, name: str = DEFAULT_LATCH_NAME) -> None:
    p = latch_path(run_dir, name=name)
    if p.exists():
        p.unlink()


def main() -> int:
    ap = argparse.ArgumentParser(description="Live kill switch latch (file-based).")
    ap.add_argument("--run-dir", required=True, help="Artifacts/run directory containing latch file.")
    ap.add_argument("--on", action="store_true", help="Engage kill switch (create latch file).")
    ap.add_argument("--off", action="store_true", help="Disengage kill switch (remove latch file).")
    ap.add_argument("--status", action="store_true", help="Print status and exit.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)

    if args.status:
        print("ENGAGED" if is_engaged(run_dir) else "DISENGAGED")
        return 0

    if args.on and args.off:
        print("ERROR: choose only one of --on/--off")
        return 2

    if args.on:
        p = engage(run_dir)
        print(f"ENGAGED: {p}")
        return 0

    if args.off:
        disengage(run_dir)
        print("DISENGAGED")
        return 0

    print("No action. Use --status, --on, or --off.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
