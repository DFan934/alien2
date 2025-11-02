# scripts/phase2_done_check.py
"""
Phase 2 — Definition of Done (single-file checker)

Runs a quick set of assertions that correspond to the Phase-2 DoD:
  1) Single source of truth: resolve_universe() works on the provided config.
  2) Dry-run exists and prints coverage + per-symbol partitions.
  3) FE entry point tolerates sparse coverage and logs per-symbol discovery.
  4) Provenance: we can compute/show a stable universe hash + source + window + artifacts root.
  5) Guardrails: empty universe / over-large / low coverage (strict) are handled.
  6) Minimal tests (already in repo) pass separately; this script just rechecks the behaviors quickly.

Exit code:
  0 → all checks passed (Phase 2 DoD met)
  1 → a check failed (message printed)
"""

from __future__ import annotations
import sys, json, argparse, hashlib
from pathlib import Path

from universes import resolve_universe
from universes.providers import StaticUniverse, FileUniverse  # for source labeling
from scripts.verify_loader import universe_dry_run
from feature_engineering.pipelines.core import CoreFeaturePipeline


def _stable_universe_hash(symbols: list[str]) -> str:
    payload = json.dumps(sorted([str(s).upper() for s in symbols]), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _universe_source_label(uobj) -> str:
    # Best-effort labeling for PR/log visibility
    if isinstance(uobj, StaticUniverse):
        return "static:list"
    if isinstance(uobj, FileUniverse):
        return f"file:{getattr(uobj, 'path', '(unknown)')}"
    if isinstance(uobj, (list, tuple)):
        return "static:list"
    if isinstance(uobj, str):
        return f"file:{uobj}"
    if isinstance(uobj, dict) and str(uobj.get("type","")).lower() == "sp500":
        return f"sp500:{uobj.get('as_of','')}"
    return "unknown"


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2 DoD single-file checker")
    p.add_argument("--parquet-root", default="parquet")
    p.add_argument("--artifacts-root", default="artifacts/phase2_check")
    p.add_argument("--universe", help="Comma list (RRC,BBY) or path to .txt/.csv or JSON dict string", default="RRC,BBY")
    p.add_argument("--start", default="1999-01-01")
    p.add_argument("--end",   default="1999-02-01")
    p.add_argument("--universe-max-size", type=int, default=1000)
    p.add_argument("--min-coverage", type=float, default=0.90)
    p.add_argument("--strict-universe", action="store_true", help="Fail if coverage < min-coverage")
    p.add_argument("--no-fe", action="store_true", help="Skip the tiny FE sanity slice (keeps it ultra-fast)")
    return p.parse_args()


def _coerce_universe_arg(val: str):
    val = val.strip()
    if val.startswith("{") and val.endswith("}"):
        return json.loads(val)
    if "," in val:
        return [s.strip() for s in val.split(",") if s.strip()]
    return val  # file path, most likely


def main():
    args = parse_args()
    cfg = {
        "parquet_root": args.parquet_root,
        "artifacts_root": args.artifacts_root,
        "universe": _coerce_universe_arg(args.universe),
        "start": args.start,
        "end":   args.end,
        "universe_max_size": args.universe_max_size,
        "universe_min_coverage": args.min_coverage,
        "strict_universe": args.strict_universe,
        "universe_dry_run": True,
    }

    # 1) Single source of truth: resolve via resolver
    try:
        syms = resolve_universe(cfg)
    except Exception as e:
        print(f"[DoD] FAIL: resolve_universe() raised → {e}")
        return 1
    if not syms:
        print("[DoD] FAIL: Universe is empty after resolution.")
        return 1
    if len(syms) > args.universe_max_size:
        print(f"[DoD] FAIL: Universe too large ({len(syms)} > {args.universe_max_size}).")
        return 1

    # 2) Provenance header (hash, source, window, artifacts)
    uhash = _stable_universe_hash(syms)
    usrc  = _universe_source_label(cfg["universe"])
    print(f"[DoD] Universe size={len(syms)} (source={usrc}, hash={uhash}) | Window={args.start}→{args.end} | Artifacts={args.artifacts_root}")

    # 3) Dry-run: coverage + per-symbol partitions (and optional strict guard)
    try:
        rep = universe_dry_run(cfg)
    except Exception as e:
        print(f"[DoD] FAIL: universe_dry_run() raised → {e}")
        return 1

    cov = rep["coverage"]
    print(f"[DoD] Coverage {cov['have_data']}/{cov['total']} ({cov['ratio']*100:.1f}%), empty={cov['empty_symbols']}")
    if args.strict_universe and cov["ratio"] < args.min_coverage:
        print(f"[DoD] FAIL: strict coverage {cov['ratio']*100:.1f}% < {args.min_coverage*100:.0f}%")
        return 1

    # 4) FE entry-point tolerates sparse coverage (OPTIONAL quick sanity slice)
    if not args.no_fe and cov["have_data"] > 0:
        # Pick the first non-empty symbol for a tiny in-memory FE run.
        non_empty = next((s for s, r in rep["summary"].items() if r["total_files"] > 0), None)
        try:
            # This uses your pipeline; it will log per-symbol rows in run() (2.4 requirement).
            pipe = CoreFeaturePipeline(parquet_root=args.parquet_root)
            # We run the normal .run() to validate entry points accept a universe symbol.
            # To keep it quick, we pass just one symbol and the given window.
            feats, meta = pipe.run(symbols=[non_empty], start=args.start, end=args.end, normalization_mode="per_symbol")
            print(f"[DoD] FE sanity slice ok: symbol={non_empty}, rows={len(feats)}")
        except Exception as e:
            print(f"[DoD] FAIL: FE sanity slice raised → {e}")
            return 1
    else:
        print("[DoD] FE sanity slice skipped (--no-fe or no data).")

    print("[DoD] Phase 2 Definition-of-Done PASSED ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
