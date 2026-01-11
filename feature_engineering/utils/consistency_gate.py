# run_gates/consistency_gate.py
from __future__ import annotations

import json
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



# ---------------------------------------------------------------------------
# Portfolio coverage gate (Task 2)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoverageGateResult:
    """Outcome of enforcing minimum bar presence on the unified clock."""

    threshold: float
    mode: str  # "abort" | "drop"
    presence_col: str
    symbol_col: str

    per_symbol_coverage: dict
    failing_symbols: list
    kept_symbols: list

    clock_len: int
    artifact_coverage_gate_path: Path
    artifact_dropped_symbols_path: Optional[Path] = None


def enforce_portfolio_bar_coverage_gate(
    bars_std: "pd.DataFrame",
    *,
    artifacts_root: Path,
    threshold: float = 0.75,
    mode: str = "abort",
    symbol_col: str = "symbol",
    presence_col: str = "bar_present",
    clock_len: Optional[int] = None,
) -> CoverageGateResult:
    """Enforce per-symbol coverage on the unified clock.

    The input is expected to already be standardized to the *unified clock*.
    Coverage is defined as mean(presence_col) per symbol.

    Artifacts written:
      - diagnostics/coverage_gate.json (always)
      - diagnostics/dropped_symbols.json (only in drop-mode when failing symbols exist)
    """
    artifacts_root = Path(artifacts_root)
    diag_dir = artifacts_root / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    if mode not in {"abort", "drop"}:
        raise ValueError(f"coverage_gate_mode must be 'abort' or 'drop' (got {mode!r})")

    if bars_std is None or len(bars_std) == 0:
        raise RuntimeError("[COVERAGE-GATE] bars_std is empty; cannot compute coverage")

    if symbol_col not in bars_std.columns:
        raise RuntimeError(f"[COVERAGE-GATE] missing symbol_col={symbol_col!r} in bars_std")
    if presence_col not in bars_std.columns:
        raise RuntimeError(f"[COVERAGE-GATE] missing presence_col={presence_col!r} in bars_std")

    # Normalize presence into numeric 0/1.
    s = bars_std[presence_col]
    if s.dtype == "bool":
        pres = s.astype("int8")
    else:
        # tolerate float/int with NaN
        pres = s.fillna(0).astype("float32")

    cov_series = pres.groupby(bars_std[symbol_col].astype(str)).mean()
    per_symbol_coverage = {str(k): float(v) for k, v in cov_series.to_dict().items()}

    failing_symbols = sorted([sym for sym, c in per_symbol_coverage.items() if float(c) < float(threshold)])
    kept_symbols = sorted([sym for sym in per_symbol_coverage.keys() if sym not in set(failing_symbols)])

    # Attempt to infer clock length if caller didn't pass it.
    if clock_len is None:
        # If fully standardized, each symbol has exactly clock_len rows.
        # We choose the max group size as a robust proxy.
        clock_len = int(bars_std.groupby(symbol_col).size().max())

    cov_path = diag_dir / "coverage_gate.json"
    payload = {
        "threshold": float(threshold),
        "mode": str(mode),
        "symbol_col": str(symbol_col),
        "presence_col": str(presence_col),
        "clock_len": int(clock_len),
        "per_symbol_coverage": per_symbol_coverage,
        "failing_symbols": failing_symbols,
        "kept_symbols": kept_symbols,
    }
    cov_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    dropped_path: Optional[Path] = None
    if failing_symbols and mode == "drop":
        dropped_path = diag_dir / "dropped_symbols.json"
        dropped_path.write_text(
            json.dumps({"dropped": failing_symbols, "kept": kept_symbols}, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    return CoverageGateResult(
        threshold=float(threshold),
        mode=str(mode),
        presence_col=str(presence_col),
        symbol_col=str(symbol_col),
        per_symbol_coverage=per_symbol_coverage,
        failing_symbols=failing_symbols,
        kept_symbols=kept_symbols,
        clock_len=int(clock_len),
        artifact_coverage_gate_path=cov_path,
        artifact_dropped_symbols_path=dropped_path,
    )
