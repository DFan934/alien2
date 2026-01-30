# data_ingestion/live/gating.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


LATCH_FILENAME = "UNATTENDED_ALLOWED"  # simple file latch in out_dir


@dataclass(frozen=True)
class GateOutcome:
    ok: bool
    reasons: list[str]
    details: Dict[str, Any]


def evaluate_promotion_gates(out_dir: Path) -> GateOutcome:
    """
    Phase 14 gates for unattended mode.

    Contract:
      - Reads out_dir/promotion_readiness.json (must exist)
      - Returns GateOutcome(ok=..., reasons=[...], details={...})
    """
    out_dir = Path(out_dir)
    readiness_path = out_dir / "promotion_readiness.json"
    reasons: list[str] = []
    details: Dict[str, Any] = {}

    if not readiness_path.exists():
        reasons.append("missing_promotion_readiness_json")
        return GateOutcome(False, reasons, details)

    try:
        obj = json.loads(readiness_path.read_text(encoding="utf-8"))
    except Exception as e:
        reasons.append(f"invalid_promotion_readiness_json:{type(e).__name__}")
        return GateOutcome(False, reasons, details)

    # We accept either style:
    #   { "gates_passed": true, ... }
    # or { "gates": { "GATE_NAME": true, ... }, ... }
    gates_passed = bool(obj.get("gates_passed", False))
    gates = obj.get("gates", {}) if isinstance(obj.get("gates"), dict) else {}

    details["promotion_readiness"] = obj

    if gates:
        # If gates dict is present, all must be True
        bad = [k for k, v in gates.items() if not bool(v)]
        if bad:
            reasons.append("failed_gates:" + ",".join(sorted(bad)))
    else:
        if not gates_passed:
            reasons.append("gates_passed_false")

    ok = len(reasons) == 0
    return GateOutcome(ok, reasons, details)


def write_promotion_readiness(out_dir: Path, readiness: Dict[str, Any]) -> Path:
    """
    Writes out_dir/promotion_readiness.json (Phase 14 artifact).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "promotion_readiness.json"

    payload = dict(readiness)
    payload.setdefault("ts_utc", datetime.now(timezone.utc).isoformat())

    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return p


def assert_unattended_allowed(out_dir: Path) -> None:
    """
    Hard gate: refuses unattended mode unless:
      1) latch file exists: out_dir/UNATTENDED_ALLOWED
      2) promotion gates pass (based on promotion_readiness.json)
    """
    out_dir = Path(out_dir)

    latch = out_dir / LATCH_FILENAME
    if not latch.exists():
        raise RuntimeError(
            f"Unattended mode blocked: missing latch file {latch.name} in {out_dir}"
        )

    outcome = evaluate_promotion_gates(out_dir)
    if not outcome.ok:
        raise RuntimeError(
            "Unattended mode blocked: promotion gates failed: "
            + ";".join(outcome.reasons)
        )
