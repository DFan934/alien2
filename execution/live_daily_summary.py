# reports/live_daily_summary.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd

from execution.cost_audit import write_cost_audit


def _read_parquet_dataset(dataset_dir: Path) -> pd.DataFrame:
    """
    Reads a dataset written as:
      <dataset_dir>/part-000001.parquet, part-000002.parquet, ...
    Returns empty DF if dir/parts don't exist.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        return pd.DataFrame()
    parts = sorted(dataset_dir.glob("part-*.parquet"))
    if not parts:
        return pd.DataFrame()
    dfs = []
    for p in parts:
        try:
            dfs.append(pd.read_parquet(p))
        except Exception:
            continue
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _write_equity_curve(run_dir: Path, *, default_equity: float = 0.0) -> pd.DataFrame:
    """
    Phase 13 artifact: run_dir/equity_curve.parquet

    Prefer positions.parquet snapshots if present (ts_utc, account_equity).
    Otherwise write a single-row curve so the artifact always exists.
    """
    run_dir = Path(run_dir)
    out_path = run_dir / "equity_curve.parquet"

    pos_df = _read_parquet_dataset(run_dir / "positions.parquet")

    if not pos_df.empty and {"ts_utc", "account_equity"}.issubset(set(pos_df.columns)):
        curve = pos_df[["ts_utc", "account_equity"]].copy()
        curve = curve.rename(columns={"account_equity": "equity"})
        curve.to_parquet(out_path, index=False)
        return curve

    # No positions snapshots → still write a valid equity curve
    curve = pd.DataFrame.from_records(
        [{"ts_utc": datetime.now(timezone.utc), "equity": float(default_equity)}]
    )
    curve.to_parquet(out_path, index=False)
    return curve


def generate_daily_summary(
    run_dir: Path,
    *,
    cfg: Optional[Dict[str, Any]] = None,
    default_equity: float = 0.0,
) -> Dict[str, Any]:
    """
    Phase 13 artifacts:
      - run_dir/daily_summary.json
      - run_dir/equity_curve.parquet
      - run_dir/cost_audit.json

    Acceptance: generates end-of-day report even if no trades occurred.
    """
    run_dir = Path(run_dir)
    cfg = cfg or {}

    attempts = _read_parquet_dataset(run_dir / "attempted_actions.parquet")
    orders = _read_parquet_dataset(run_dir / "orders.parquet")
    fills = _read_parquet_dataset(run_dir / "fills.parquet")

    # Trades: if you have a consolidated trades.parquet (from backtest or other),
    # we’ll use it. Otherwise, cost audit will be empty-but-valid.
    trades_path = run_dir / "trades.parquet"
    if trades_path.exists():
        try:
            trades_df = pd.read_parquet(trades_path)
        except Exception:
            trades_df = pd.DataFrame()
    else:
        trades_df = pd.DataFrame()

    # Always write the cost audit, even if empty
    write_cost_audit(trades_df, cfg, run_dir, sample_n=int(cfg.get("cost_audit_sample_n", 25)))

    # Always write equity curve
    curve = _write_equity_curve(run_dir, default_equity=float(cfg.get("equity", default_equity) or default_equity))
    ending_equity = float(curve["equity"].iloc[-1]) if not curve.empty and "equity" in curve.columns else float(default_equity)

    # Derive daily counts
    n_attempts = int(len(attempts)) if not attempts.empty else 0
    n_blocked = int((attempts.get("allowed") == False).sum()) if (not attempts.empty and "allowed" in attempts.columns) else 0  # noqa: E712
    n_allowed = n_attempts - n_blocked
    n_orders = int(len(orders)) if not orders.empty else 0
    n_fills = int(len(fills)) if not fills.empty else 0

    summary = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "attempted_actions": n_attempts,
            "allowed_actions": n_allowed,
            "blocked_actions": n_blocked,
            "orders_submitted": n_orders,
            "fills": n_fills,
        },
        "ending_equity": ending_equity,
        "artifacts": {
            "cost_audit_json": "cost_audit.json",
            "daily_summary_json": "daily_summary.json",
            "equity_curve_parquet": "equity_curve.parquet",
        },
        "note": "Generated daily summary (safe even with zero trades).",
    }

    _write_json(run_dir / "daily_summary.json", summary)
    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Phase 13: live daily summary + cost audit")
    ap.add_argument("--run-dir", required=True, help="Directory containing live artifacts (orders/fills/positions/attempts)")
    ap.add_argument("--equity", type=float, default=0.0, help="Default equity if no positions snapshots exist")
    args = ap.parse_args()

    generate_daily_summary(Path(args.run_dir), cfg={"equity": args.equity}, default_equity=args.equity)
