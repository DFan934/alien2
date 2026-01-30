# execution/cost_audit.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def _cfg_bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "t", "yes", "y", "on"):
            return True
        if s in ("0", "false", "f", "no", "n", "off"):
            return False
    return default


def _cfg_float(v: Any, default: float) -> float:
    if v is None:
        return float(default)
    if isinstance(v, (int, float, np.floating)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("none", "null", ""):
            return float(default)
        try:
            return float(s)
        except Exception:
            return float(default)
    return float(default)


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def write_cost_audit(
    trades_df: Optional[pd.DataFrame],
    cfg: Dict[str, Any],
    out_dir: Path,
    *,
    sample_n: int = 25,
) -> Dict[str, Any]:
    """
    Phase 13 artifact: out_dir/cost_audit.json

    This is a *proof* artifact: records raw cfg knobs + effective numeric knobs,
    and reconciles modeled vs realized fields if present.

    Works even if trades_df is None/empty (writes an empty-but-valid audit).
    """
    out_dir = Path(out_dir)
    out_path = out_dir / "cost_audit.json"

    debug_no_costs_effective = _cfg_bool(cfg.get("debug_no_costs", False), default=False)

    spread_bp_eff = _cfg_float(cfg.get("spread_bp", 1.0), default=1.0)
    slippage_bp_eff = _cfg_float(cfg.get("slippage_bp", 0.0), default=0.0)
    commission_eff = _cfg_float(cfg.get("commission", 0.0), default=0.0)
    impact_eff = _cfg_float(cfg.get("impact_bps_per_adv_frac", 25.0), default=25.0)

    if trades_df is None or len(trades_df) == 0:
        audit = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "debug_no_costs_raw": cfg.get("debug_no_costs", None),
            "debug_no_costs_effective": debug_no_costs_effective,
            "raw_cfg": {
                "spread_bp": cfg.get("spread_bp", None),
                "slippage_bp": cfg.get("slippage_bp", None),
                "commission": cfg.get("commission", None),
                "impact_bps_per_adv_frac": cfg.get("impact_bps_per_adv_frac", None),
            },
            "effective_knobs": {
                "spread_bp": spread_bp_eff,
                "slippage_bp": slippage_bp_eff,
                "commission": commission_eff,
                "impact_bps_per_adv_frac": impact_eff,
            },
            "n_trades": 0,
            "totals": {
                "sum_modeled_cost_total": 0.0,
                "sum_realized_pnl": 0.0,
                "sum_realized_pnl_after_costs": 0.0,
            },
            "sample": [],
            "note": "No trades present; wrote empty-but-valid audit.",
        }
        _write_json(out_path, audit)
        return audit

    df = trades_df.copy()

    qty_raw = pd.to_numeric(df.get("qty", 0.0), errors="coerce").fillna(0.0)
    qty = qty_raw.abs()

    entry = pd.to_numeric(df.get("entry_price", 0.0), errors="coerce").fillna(0.0)
    exit_ = pd.to_numeric(df.get("exit_price", 0.0), errors="coerce").fillna(0.0)
    mid = (entry + exit_) / 2.0

    if "half_spread_usd" in df.columns:
        half_spread_usd = pd.to_numeric(df["half_spread_usd"], errors="coerce")
    else:
        half_spread_usd = pd.Series(np.nan, index=df.index, dtype=float)

    fallback_half_spread = (spread_bp_eff / 1e4) * mid
    half_spread_usd_used = half_spread_usd.where(~half_spread_usd.isna(), fallback_half_spread).fillna(0.0)

    if "adv_frac" in df.columns:
        adv_frac = pd.to_numeric(df["adv_frac"], errors="coerce").fillna(0.0)
    else:
        adv_frac = pd.Series(0.0, index=df.index, dtype=float)

    # Component breakdown (aligned with your backtest diagnostic logic)
    spread_cost = 2.0 * half_spread_usd_used * qty
    commission_cost = 2.0 * commission_eff * qty
    slippage_cost = (slippage_bp_eff / 1e4) * mid * 2.0 * qty
    impact_cost = (impact_eff / 1e4) * mid * adv_frac * qty

    modeled_cost_total = pd.to_numeric(df.get("modeled_cost_total", 0.0), errors="coerce").fillna(0.0)
    realized_pnl = pd.to_numeric(df.get("realized_pnl", 0.0), errors="coerce").fillna(0.0)
    realized_after = pd.to_numeric(
        df.get("realized_pnl_after_costs", realized_pnl - modeled_cost_total),
        errors="coerce",
    ).fillna(0.0)

    sort_col = "entry_ts" if "entry_ts" in df.columns else None
    dfx = df.copy()
    dfx["_mid"] = mid
    dfx["_half_spread_usd_used"] = half_spread_usd_used
    dfx["_spread_cost"] = spread_cost
    dfx["_commission_cost"] = commission_cost
    dfx["_slippage_cost"] = slippage_cost
    dfx["_impact_cost"] = impact_cost
    dfx["_modeled_cost_total"] = modeled_cost_total
    dfx["_realized_pnl"] = realized_pnl
    dfx["_realized_after"] = realized_after

    if sort_col:
        dfx = dfx.sort_values(sort_col)

    keep_cols = [c for c in ("symbol", "entry_ts", "exit_ts", "qty", "entry_price", "exit_price") if c in dfx.columns]
    keep_cols += [
        "_mid",
        "_half_spread_usd_used",
        "_spread_cost",
        "_commission_cost",
        "_slippage_cost",
        "_impact_cost",
        "_modeled_cost_total",
        "_realized_pnl",
        "_realized_after",
    ]
    sample_df = dfx[keep_cols].head(int(sample_n)).copy()

    def _round(v: Any) -> Any:
        try:
            if isinstance(v, (int, float, np.floating)):
                return float(round(float(v), 10))
        except Exception:
            pass
        return v

    sample = [{k: _round(r[k]) for k in sample_df.columns} for _, r in sample_df.iterrows()]

    audit = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "debug_no_costs_raw": cfg.get("debug_no_costs", None),
        "debug_no_costs_effective": debug_no_costs_effective,
        "raw_cfg": {
            "spread_bp": cfg.get("spread_bp", None),
            "slippage_bp": cfg.get("slippage_bp", None),
            "commission": cfg.get("commission", None),
            "impact_bps_per_adv_frac": cfg.get("impact_bps_per_adv_frac", None),
        },
        "effective_knobs": {
            "spread_bp": spread_bp_eff,
            "slippage_bp": slippage_bp_eff,
            "commission": commission_eff,
            "impact_bps_per_adv_frac": impact_eff,
        },
        "n_trades": int(len(df)),
        "totals": {
            "sum_modeled_cost_total": float(modeled_cost_total.sum()),
            "sum_realized_pnl": float(realized_pnl.sum()),
            "sum_realized_pnl_after_costs": float(realized_after.sum()),
        },
        "sample": sample,
    }

    _write_json(out_path, audit)
    return audit
