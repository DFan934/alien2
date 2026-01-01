# run_gates/consistency_gate.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ConsistencyGateInputs:
    run_id: str
    artifacts_root: Path

    # what consolidation actually observed/did
    decisions_paths_found: int
    consolidated_decisions_written: bool
    consolidated_decisions_rows: int

    # what report phase claimed
    report_phase_decisions_count: int
    report_said_no_decisions: bool

    # debugging / auditability
    searched_dirs: Tuple[Path, ...]
    decision_files_found: Tuple[Path, ...] = tuple()
    consolidated_path: Optional[Path] = None
    report_path: Optional[Path] = None


def _fmt_paths(paths: Iterable[Path], limit: int = 80) -> str:
    p_list = list(paths)
    head = p_list[:limit]
    lines = "\n".join(f"  - {str(p)}" for p in head)
    if len(p_list) > limit:
        lines += f"\n  ... ({len(p_list)-limit} more)"
    return lines


def run_consistency_gate(inp: ConsistencyGateInputs) -> None:
    """
    Hard-fail if the run contradicts itself.

    Primary rule:
      - If report says “no decisions” but consolidation wrote decisions (or has rows), fail.
      - If report claims decisions but consolidation has none, fail.
    """
    # Normalize “has decisions” concepts
    consolidation_has_any = (
        inp.decisions_paths_found > 0
        or inp.consolidated_decisions_written
        or inp.consolidated_decisions_rows > 0
    )
    report_has_any = (inp.report_phase_decisions_count > 0) and (not inp.report_said_no_decisions)

    contradictions = []

    # Case A: report says none, but consolidation indicates some
    if inp.report_said_no_decisions and consolidation_has_any:
        contradictions.append(
            "Report phase claimed NO decisions, but consolidation indicates decisions exist."
        )

    # Case B: report claims decisions, but consolidation indicates none
    if report_has_any and (not consolidation_has_any):
        contradictions.append(
            "Report phase claimed decisions exist, but consolidation indicates NO decisions."
        )

    # Case C: report count > 0 but report also says no decisions (internal contradiction)
    if inp.report_phase_decisions_count > 0 and inp.report_said_no_decisions:
        contradictions.append(
            "Report is internally inconsistent: report_phase_decisions_count > 0 but report_said_no_decisions=True."
        )

    if contradictions:
        debug = []
        debug.append("=== CONSISTENCY GATE FAILED ===")
        debug.append(f"run_id: {inp.run_id}")
        debug.append(f"artifacts_root: {str(inp.artifacts_root)}")
        debug.append("")
        debug.append("Searched dirs:")
        debug.append(_fmt_paths(inp.searched_dirs))
        debug.append("")
        debug.append(f"decisions_paths_found: {inp.decisions_paths_found}")
        debug.append(f"consolidated_decisions_written: {inp.consolidated_decisions_written}")
        debug.append(f"consolidated_decisions_rows: {inp.consolidated_decisions_rows}")
        debug.append(f"report_phase_decisions_count: {inp.report_phase_decisions_count}")
        debug.append(f"report_said_no_decisions: {inp.report_said_no_decisions}")
        if inp.consolidated_path is not None:
            debug.append(f"consolidated_path: {str(inp.consolidated_path)}")
        if inp.report_path is not None:
            debug.append(f"report_path: {str(inp.report_path)}")
        debug.append("")
        if inp.decision_files_found:
            debug.append("Decision files found:")
            debug.append(_fmt_paths(inp.decision_files_found))
            debug.append("")
        debug.append("Contradictions:")
        for c in contradictions:
            debug.append(f"- {c}")

        raise RuntimeError("\n".join(debug))
