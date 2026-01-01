# reporting/walkforward_report.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, List

import pandas as pd


@dataclass(frozen=True)
class ReportResult:
    run_id: str
    artifacts_root: Path
    searched_dirs: Tuple[Path, ...]
    consolidated_decisions_path: Optional[Path]
    decisions_count_seen: int
    report_path: Path
    report_had_no_decisions_banner: bool


def _ensure_parent_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _load_consolidated_decisions(path: Path) -> pd.DataFrame:
    # We assume parquet for consolidated; if you want csv too, expand here.
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() != ".parquet":
        raise ValueError(f"Expected consolidated decisions parquet, got: {path}")
    return pd.read_parquet(path)


def _format_walkforward_summary(df: pd.DataFrame) -> str:
    """
    Customize this to your system’s actual columns and walk-forward layout.
    This produces a readable markdown summary that is stable even if
    some optional columns are missing.
    """
    lines: List[str] = []
    lines.append("# Walk-Forward Report")
    lines.append("")
    lines.append(f"- Total decisions rows: **{len(df)}**")

    # Common columns you might have:
    # ts, symbol, action/side, ev, prob, regime, split_id/window_id, etc.
    colset = set(df.columns)

    def safe_count(col: str) -> Optional[int]:
        if col in colset:
            return int(df[col].notna().sum())
        return None

    if "symbol" in colset:
        lines.append(f"- Unique symbols: **{df['symbol'].nunique()}**")

    for c in ["ts", "action", "side", "regime", "split_id", "window_id", "ev", "prob"]:
        cnt = safe_count(c)
        if cnt is not None:
            lines.append(f"- Non-null `{c}`: **{cnt}**")

    lines.append("")
    lines.append("## Quick Tables")
    lines.append("")

    # By symbol
    if "symbol" in colset:
        sym = df["symbol"].value_counts().head(25)
        lines.append("### Decisions by Symbol (top 25)")
        lines.append("")
        lines.append(sym.to_frame("count").to_markdown())
        lines.append("")

    # By action/side
    if "action" in colset:
        act = df["action"].value_counts().head(25)
        lines.append("### Decisions by Action (top 25)")
        lines.append("")
        lines.append(act.to_frame("count").to_markdown())
        lines.append("")

    # By walk-forward window if present
    wf_key = None
    for k in ("split_id", "window_id", "wf_window", "train_end", "test_start"):
        if k in colset:
            wf_key = k
            break
    if wf_key is not None:
        wv = df[wf_key].value_counts().head(50)
        lines.append(f"### Decisions by `{wf_key}` (top 50)")
        lines.append("")
        lines.append(wv.to_frame("count").to_markdown())
        lines.append("")

    return "\n".join(lines)


def generate_walkforward_report(
    *,
    run_id: str,
    artifacts_root: Path,
    searched_dirs: Sequence[Path],
    consolidated_decisions_relpath: str = "consolidated/decisions_consolidated.parquet",
    report_relpath: str = "reports/walkforward_report.md",
    verbose: bool = True,
) -> ReportResult:
    """
    Generate a walk-forward summary report.

    Key guarantee:
      - It only emits “NO DECISIONS FOUND” banner if decisions_count_seen == 0
        AND the consolidated file does not exist or is empty.
    """
    artifacts_root = artifacts_root.expanduser().resolve()
    dirs = tuple(Path(d).expanduser().resolve() for d in searched_dirs)

    consolidated_path = (artifacts_root / consolidated_decisions_relpath).resolve()
    df = _load_consolidated_decisions(consolidated_path) if consolidated_path.exists() else pd.DataFrame()

    decisions_count = int(len(df))
    no_decisions_banner = (decisions_count == 0)

    report_path = (artifacts_root / report_relpath).resolve()
    _ensure_parent_dir(report_path)

    if verbose:
        print("[Report] run_id:", run_id)
        print("[Report] artifacts_root:", str(artifacts_root))
        print("[Report] searched_dirs:")
        for d in dirs:
            print("  -", str(d))
        print("[Report] consolidated_decisions_path:", str(consolidated_path))
        print("[Report] decisions_count_seen:", decisions_count)

    if no_decisions_banner:
        content = "\n".join(
            [
                "# Walk-Forward Report",
                "",
                "## NO DECISIONS FOUND",
                "",
                "The consolidated decisions file was missing or empty.",
                f"- Expected consolidated path: `{consolidated_path}`",
                "",
            ]
        )
    else:
        content = _format_walkforward_summary(df)

    report_path.write_text(content, encoding="utf-8")

    return ReportResult(
        run_id=run_id,
        artifacts_root=artifacts_root,
        searched_dirs=dirs,
        consolidated_decisions_path=consolidated_path if consolidated_path.exists() else None,
        decisions_count_seen=decisions_count,
        report_path=report_path,
        report_had_no_decisions_banner=no_decisions_banner,
    )
