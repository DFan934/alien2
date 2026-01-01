# scripts/a2_report.py


from __future__ import annotations

import json
from pathlib import Path

from feature_engineering.utils.artifacts_root import resolve_artifacts_root


from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

from pathlib import Path

def _resolve_path(path_like: str | Path) -> Path:
    """Resolve paths relative to CWD or the repo root (parent of scripts/)."""
    p = Path(path_like).expanduser()
    if p.exists():
        return p.resolve()
    repo_root = Path(__file__).parents[1]  # repo root if this file is scripts/a2_report.py
    alt = (repo_root / str(path_like)).expanduser()
    if alt.exists():
        return alt.resolve()
    # Helpful error with context
    raise FileNotFoundError(
        f"Could not find {path_like!r}. "
        f"CWD={Path.cwd()}  tried={p} and {alt}"
    )


def _load_all_decisions(artifacts_root: Path) -> pd.DataFrame:
    """Concatenate fold decisions; tolerate empty folds."""
    root = Path(artifacts_root)
    rows = []
    for fold_dir in sorted(root.glob("fold_*")):
        dec_path = fold_dir / "decisions.parquet"
        if dec_path.exists() and dec_path.stat().st_size > 0:
            df = pd.read_parquet(dec_path)
            df["fold"] = fold_dir.name
            rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["timestamp","symbol","mu","p","gate","label","fold"])
    out = pd.concat(rows, ignore_index=True)
    # normalize dtypes
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=False)
    out["gate"] = out["gate"].astype(int)
    out["label"] = out["label"].astype(int)
    return out.sort_values("timestamp").reset_index(drop=True)

def _load_all_probs(artifacts_root: Path) -> pd.DataFrame:
    """
    Optional: read fold_*_probs.csv written by runner (if present).
    Expects columns: p_raw, p_cal (optional), y, is_entry, and ideally timestamp.
    Returns a single DataFrame with a 'fold' column.
    """
    root = Path(artifacts_root)
    rows = []
    for fold_dir in sorted(root.glob("fold_*")):
        for name in (f"{fold_dir.name}_probs.csv", "fold_probs.csv", "probs.csv"):
            pth = fold_dir / name
            if pth.exists() and pth.stat().st_size > 0:
                df = pd.read_csv(pth)
                df["fold"] = fold_dir.name
                # normalize types if timestamp exists
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False, errors="coerce")
                rows.append(df)
                break
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)

def _attach_probs_to_decisions(dec: pd.DataFrame, probs: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer merge by (fold, timestamp) if possible; otherwise align by position per-fold.
    Returns a copy of 'dec' with optional p_raw/p_cal columns added.
    """
    if probs.empty or dec.empty:
        return dec.copy()

    dec2 = dec.copy()
    if "timestamp" in probs.columns:
        on_cols = ["fold", "timestamp"]
        keep = ["p_raw", "p_cal", "is_entry", "y"]
        keep = [c for c in keep if c in probs.columns]
        m = dec2.merge(probs[["fold","timestamp"] + keep], on=on_cols, how="left", suffixes=("",""))
        return m

    # Fallback: per-fold positional alignment
    out = []
    for f, g in dec2.groupby("fold", sort=False):
        gp = probs[probs["fold"] == f]
        g = g.reset_index(drop=True)
        gp = gp.reset_index(drop=True)
        if len(g) == len(gp):
            # copy through aligned columns
            for c in ("p_raw","p_cal","is_entry","y"):
                if c in gp.columns:
                    g[c] = gp[c].values
        else:
            # lengths differ → leave untouched for this fold
            pass
        out.append(g)
    return pd.concat(out, ignore_index=True)


def _attach_nextbar_returns(decisions: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    """Attach next-bar open/close and compute 1-bar return for each decision row timestamp."""
    px = pd.read_csv(csv_path)
    # Expect columns: Date, Time, Open, High, Low, Close, Volume
    px["timestamp"] = pd.to_datetime(px["Date"] + " " + px["Time"])
    px = px.sort_values("timestamp").reset_index(drop=True)
    # next bar ret = next_close/next_open - 1.0
    next_open  = px["Open"].shift(-1)
    next_close = px["Close"].shift(-1)
    ret1 = (next_close / next_open - 1.0).fillna(0.0)
    # Map by timestamp
    r_by_ts = pd.Series(ret1.values, index=px["timestamp"].values)
    # Join into decisions
    dec = decisions.copy()
    dec["ret1"] = dec["timestamp"].map(r_by_ts).fillna(0.0)
    return dec

def _max_drawdown(curve: pd.Series) -> float:
    """Return max drawdown (as a negative fraction)."""
    roll_max = curve.cummax()
    dd = curve/roll_max - 1.0
    return float(dd.min())

def _rolling_sharpe(returns: pd.Series, window: int = 20) -> pd.Series:
    """Rolling Sharpe over entry sequence (not time)."""
    mu = returns.rolling(window).mean()
    sd = returns.rolling(window).std(ddof=1)
    sharpe = mu / sd.replace(0, np.nan)
    return sharpe

def _calibration_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"p": p, "y": y})
    # small-n guard: min(bins, unique p - 1)
    q = min(bins, max(1, df["p"].nunique() - 1))
    try:
        df["bin"] = pd.qcut(df["p"], q=q, duplicates="drop")
    except Exception:
        return pd.DataFrame(columns=["bin","p_mean","y_rate","n"])
    rel = (
        df.groupby("bin", observed=True)
          .agg(p_mean=("p","mean"), y_rate=("y","mean"), n=("p","size"))
          .reset_index()
    )
    rel["bin"] = rel["bin"].astype(str)
    return rel

def generate_report(*, artifacts_root: str | Path, csv_path: str | Path, out_dir: str | Path | None = None) -> Dict:
    """
    Read A2 fold outputs and print a rich console summary (no leakage).
    - Uses 'gate' to decide entries.
    - PnL model: 1 unit per entry, next-bar open→close return, zero costs.
    """



    #artifacts_root = Path(artifacts_root)

    if isinstance(artifacts_root, dict):
        artifacts_root = resolve_artifacts_root(artifacts_root, create=False)
    else:
        artifacts_root = Path(artifacts_root)

    print(f"[RunContext] artifacts_root={artifacts_root}")

    csv_path = Path(csv_path)
    if out_dir is None:
        out_dir = artifacts_root
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


    # --- NEW: try to read artifacts meta (for label_horizon) ---
    meta_horizon = "UNKNOWN"
    try:

        meta_path = (Path(artifacts_root) / "meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            meta_horizon = str(meta.get("label_horizon", "UNKNOWN"))
            meta_side = meta.get("label_side", "UNKNOWN")
            meta_thresh = meta.get("label_threshold", "UNKNOWN")
            print(f"[Report] label_horizon = {meta_horizon}")
            print(f"[Report] label_side    = {meta_side}   label_threshold = {meta_thresh}")

    except Exception:
        pass

    dec = _load_all_decisions(artifacts_root)
    # Try to load optional per-fold probs (p_raw/p_cal/is_entry) and attach
    probs = _load_all_probs(artifacts_root)
    dec = _attach_probs_to_decisions(dec, probs)

    # --- NEW: p-spread telemetry on all scanned rows ---
    '''p_min = float("nan")
    p_max = float("nan")
    p_iqr = float("nan")'''

    prob_source = "p"
    if "p_cal" in dec.columns and dec["p_cal"].notna().any():
        prob_source = "p_cal"
    elif "p_raw" in dec.columns and dec["p_raw"].notna().any():
        prob_source = "p_raw"

    # --- p-spread telemetry on all scanned rows (using chosen source) ---
    p_min = p_max = p_iqr = float("nan")
    if not dec.empty and prob_source in dec.columns and dec[prob_source].notna().any():
        p_vals = dec[prob_source].astype(float).to_numpy()
        p_min = float(np.nanmin(p_vals))
        p_max = float(np.nanmax(p_vals))
        try:
            q75, q25 = np.nanpercentile(p_vals, [75, 25])
            p_iqr = float(q75 - q25)
        except Exception:
            pass


    if dec.empty:
        print("A2-Report: no decisions found.")
        return {"n_test": 0, "n_entries": 0}

    # Attach returns
    dec = _attach_nextbar_returns(dec, csv_path)

    # AUCs
    auc_all = float("nan")
    auc_entries = float("nan")
    try:
        if dec["label"].nunique() > 1 and prob_source in dec.columns:
            auc_all = float(roc_auc_score(dec["label"].to_numpy(), dec[prob_source].to_numpy()))

    except Exception:
        pass
    try:
        ent = dec[dec["gate"] == 1]
        if len(ent) > 0 and ent["label"].nunique() > 1:
            auc_entries = float(roc_auc_score(ent["label"].to_numpy(), ent[prob_source].to_numpy()))
    except Exception:
        pass

    # --- Flipped AUCs (sign check) ---
    auc_all_flip = float("nan")
    auc_entries_flip = float("nan")
    try:
        if dec["label"].nunique() > 1:
            #p_all = dec["p"].to_numpy(dtype=float)
            p_all = dec[prob_source].to_numpy(dtype=float)

            y_all = dec["label"].to_numpy(dtype=float)
            auc_all_flip = float(roc_auc_score(y_all, 1.0 - p_all))
    except Exception:
        pass
    try:
        if len(ent) > 0 and ent["label"].nunique() > 1:
            #p_ent = ent["p"].to_numpy(dtype=float)
            p_ent = ent[prob_source].to_numpy(dtype=float)

            y_ent = ent["label"].to_numpy(dtype=float)
            auc_entries_flip = float(roc_auc_score(y_ent, 1.0 - p_ent))
    except Exception:
        pass

    # “1 unit per entry” equity curve over the entry sequence (not calendar time)
    entries = dec[dec["gate"] == 1].copy()
    entries["pnl"] = entries["ret1"]  # 1-unit, zero-cost
    if not entries.empty:
        equity = (1.0 + entries["pnl"]).cumprod()
        max_dd = _max_drawdown(equity)
        rsh = _rolling_sharpe(entries["pnl"], window=20).describe()
    else:
        equity = pd.Series(dtype=float)
        max_dd = float("nan")
        rsh = pd.Series(dtype=float)

    # Win-rate & trade stats
    win_rate = float((entries["pnl"] > 0).mean()) if not entries.empty else float("nan")
    avg_ret  = float(entries["pnl"].mean()) if not entries.empty else float("nan")

    # Calibration table (all scanned rows)
    cal = _calibration_table(dec["p"].to_numpy(), dec["label"].to_numpy(), bins=10)

    # Persist handy artifacts
    # 1) trade-level histogram CSV
    hist_path = out_dir / "trade_pnl_hist.csv"
    if not entries.empty:
        entries["pnl"].to_frame("pnl").to_csv(hist_path, index=False)
        print(f"[Saved] trade-pnl histogram data → {hist_path}")
    # 2) write summary JSON
    summary = {
        "n_test": int(len(dec)),
        "n_entries": int(len(entries)),
        "auc_all": auc_all,
        "auc_entries": auc_entries,
        "win_rate": win_rate,
        "avg_ret": avg_ret,
        "max_drawdown": max_dd,
        "rolling20_sharpe": {k: (None if pd.isna(v) else float(v)) for k, v in rsh.to_dict().items()},
        "label_horizon": meta_horizon,  # NEW
        "p_min": p_min,
        "p_max": p_max,
        "p_iqr": p_iqr,

    }

    (out_dir / "a2_report.json").write_text(json.dumps(summary, indent=2))
    # 3) calibration deciles CSV
    if not cal.empty:
        cal.to_csv(out_dir / "calibration_deciles.csv", index=False)

    # Console summary (styled like your older run_backtest)
    print(f"\n[Report] label_horizon = {meta_horizon}")


    print("\n=== Trade-level P&L Summary ===")
    if len(entries) > 0:
        print(f"count     {len(entries):.0f}")
        print(f"mean      {avg_ret:.6f}")
        print(f"std       {float(entries['pnl'].std(ddof=1)) if len(entries)>1 else float('nan'):.6f}")
        print(f"min       {float(entries['pnl'].min()):.6f}")
        print(f"25%       {float(entries['pnl'].quantile(0.25)):.6f}")
        print(f"50%       {float(entries['pnl'].quantile(0.50)):.6f}")
        print(f"75%       {float(entries['pnl'].quantile(0.75)):.6f}")
        print(f"max       {float(entries['pnl'].max()):.6f}")
        print(f"Win-rate = {win_rate:.3f}")
    else:
        print("No entries taken.")

    print("\n=== ROC AUC (all vs entries) ===")


    print(f"AUC(all)={auc_all if not np.isnan(auc_all) else 'nan'}")
    print(f"AUC(entries)={auc_entries if not np.isnan(auc_entries) else 'nan'}")

    print(f"AUC(all, 1-p)={auc_all_flip if not np.isnan(auc_all_flip) else 'nan'}")
    print(f"AUC(entries, 1-p)={auc_entries_flip if not np.isnan(auc_entries_flip) else 'nan'}")

    print(f"(p_min={p_min:.3f}  p_max={p_max:.3f}  p_IQR={p_iqr:.3f})")  # NEW

    # --- Reliability / Brier diagnostics (raw p) ---
    '''try:
        p_all = dec["p"].to_numpy(dtype=float)
        y_all = dec["label"].to_numpy(dtype=float)
        print_reliability("raw (all)", p_all, y_all)
    except Exception:
        pass

    try:
        if len(ent) > 0:
            p_ent = ent["p"].to_numpy(dtype=float)
            y_ent = ent["label"].to_numpy(dtype=float)
            print_reliability("raw (entries)", p_ent, y_ent)
    except Exception:
        pass'''

    # --- Reliability / Brier diagnostics (show raw & calibrated if available) ---
    y_all = dec["label"].to_numpy(dtype=float)
    y_ent = ent["label"].to_numpy(dtype=float) if len(ent) > 0 else np.array([])

    if "p_raw" in dec.columns and dec["p_raw"].notna().any():
        print_reliability("raw (all)", dec["p_raw"].to_numpy(dtype=float), y_all)
        if len(ent) > 0:
            print_reliability("raw (entries)", ent["p_raw"].to_numpy(dtype=float), y_ent)

    if "p_cal" in dec.columns and dec["p_cal"].notna().any():
        print_reliability("calibrated (all)", dec["p_cal"].to_numpy(dtype=float), y_all)
        if len(ent) > 0:
            print_reliability("calibrated (entries)", ent["p_cal"].to_numpy(dtype=float), y_ent)

    print("\n=== Drawdown ===")
    print(f"Max drawdown = {max_dd*100:.2f}%")

    print("\n=== Rolling 20-entry Sharpe ===")
    print(rsh if not rsh.empty else "n/a")

    print("\n=== Calibration (deciles) ===")
    if not cal.empty:
        print(cal[["p_mean","y_rate","n"]].to_string(index=False))
    else:
        print("n/a (insufficient variety)")

    return {"decisions": dec, "entries": entries, "summary": summary}


def _brier_score(p: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(p) & np.isfinite(y)
    if m.sum() == 0: return float('nan')
    return float(np.mean((p[m] - y[m])**2))

def _reliability_table(p: np.ndarray, y: np.ndarray, bins: int = 10):
    # returns list of (p_mean, y_rate, n) for evenly spaced probability bins
    p = np.asarray(p); y = np.asarray(y)
    m = np.isfinite(p) & np.isfinite(y)
    p, y = p[m], y[m]
    if len(p) == 0: return []
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for i in range(bins):
        lo, hi = edges[i], edges[i+1]
        sel = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        if sel.sum() == 0:
            out.append((float((lo+hi)/2), float('nan'), 0))
        else:
            out.append((float(p[sel].mean()), float(y[sel].mean()), int(sel.sum())))
    return out

def print_reliability(name: str, p: np.ndarray, y: np.ndarray):
    print(f"\n=== Calibration ({name}) ===")
    print(f"Brier = {_brier_score(p, y):.6f}")
    rows = _reliability_table(p, y, bins=10)
    print("  p_mean   y_rate    n")
    for pm, yr, n in rows:
        pm_s = f"{pm:.6f}" if np.isfinite(pm) else "nan"
        yr_s = f"{yr:.6f}" if np.isfinite(yr) else "nan"
        print(f"{pm_s:>8} {yr_s:>8} {n:4d}")


# ------------------------------
# 5.5: Fold aggregator & promotion gates
# ------------------------------
from dataclasses import dataclass, asdict
from scipy.stats import spearmanr  # if available; otherwise small fallback

@dataclass(frozen=True)
class GateThresholds:
    # Bucket-level gates (per regime x side)
    analog_spearman_min: float = 0.30
    ece_max: float = 0.03                  # 3%
    # For the Brier “≤ dev + 0.01” gate, we accept a dev_brier input; if None, gate is marked UNKNOWN.
    brier_additive_worst: float = 0.01
    # Portfolio gates
    sharpe_min: float = 1.2
    mdd_max_abs: float = 0.12              # 12%
    # Fallback
    fallback_rate_max: float = 0.15
    # Costs sanity (bps)
    slippage_med_abs_bps_max: float = 10.0
    slippage_p99_abs_bps_max: float = 25.0
    # Rolling “not contradict” allowance
    delta_pf_ok: float = 0.20
    delta_sharpe_ok: float = 0.20

def _find_fold_dirs(artifacts_root: Path) -> list[Path]:
    return sorted([p for p in Path(artifacts_root).glob("fold_*") if p.is_dir()])

def _load_fold_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def _load_fold_metrics(artifacts_root: Path) -> list[dict]:
    rows = []
    for fd in _find_fold_dirs(artifacts_root):
        fm = fd / "metrics" / "fold_metrics.json"
        if fm.exists():
            payload = _load_fold_json(fm)
            if payload is not None:
                rows.append({"fold": fd.name, **payload})
    return rows

def _concat_trades(artifacts_root: Path) -> pd.DataFrame:
    out = []
    for fd in _find_fold_dirs(artifacts_root):
        tp = fd / "trades.parquet"
        if tp.exists() and tp.stat().st_size > 0:
            try:
                df = pd.read_parquet(tp)
                df["fold"] = fd.name
                out.append(df)
            except Exception:
                pass
    if not out:
        return pd.DataFrame(columns=["entry_ts","exit_ts","qty","realized_pnl_after_costs","fold"])
    df = pd.concat(out, ignore_index=True)
    # normalize timestamps
    for col in ("entry_ts","exit_ts"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def _portfolio_metrics_from_trades(trades: pd.DataFrame) -> dict:
    """
    Compute portfolio PF/Sharpe/MDD from realized PnL series (naive but consistent).
    Assumes trades already net of costs if 'realized_pnl_after_costs' exists.
    """
    if trades.empty:
        return {
            "n_trades": 0, "net_pnl": 0.0, "sharpe": float("nan"),
            "max_drawdown": float("nan"), "turnover": 0.0, "exposure_proxy": 0.0
        }

    pnl = trades.get("realized_pnl_after_costs", trades.get("realized_pnl", pd.Series(dtype=float))).astype(float)
    # naive “per-trade return” proxy for Sharpe
    mu = pnl.mean()
    sd = pnl.std(ddof=1)
    sharpe = float(mu / sd) if sd > 0 else float("nan")
    # equity curve (per trade)
    eq = (1.0 + pnl.fillna(0.0)).cumprod()
    mdd = _max_drawdown(eq)

    # crude turnover/exposure proxies (don’t block promotion if missing)
    turnover = float(len(trades))
    # exposure proxy: average concurrent open positions from overlapping intervals (approx)
    exposure_proxy = float(turnover)  # cheap placeholder; replace with real overlap calc when available

    # slippage sanity if columns present
    med_abs_bps = p99_abs_bps = None
    if "slippage_model_bps" in trades.columns and "slippage_realized_bps" in trades.columns:
        err = (trades["slippage_model_bps"] - trades["slippage_realized_bps"]).abs()
        med_abs_bps = float(err.median())
        p99_abs_bps = float(err.quantile(0.99))

    return {
        "n_trades": int(len(trades)),
        "net_pnl": float(pnl.sum()),
        "sharpe": float(sharpe),
        "max_drawdown": float(mdd),
        "turnover": turnover,
        "exposure_proxy": exposure_proxy,
        "slippage_med_abs_bps": med_abs_bps,
        "slippage_p99_abs_bps": p99_abs_bps,
    }

def _bucket_gate_results(fold_metrics: list[dict], thresholds: GateThresholds, dev_brier: float | None) -> dict:
    """
    Aggregate bucket-level gates from per-fold metrics payloads.
    We combine folds by simple mean where applicable.
    """
    # Gather by regime x side
    bucket = {}
    # Collect analog fidelity across folds
    for row in fold_metrics:
        af = row.get("analog_fidelity_by_bucket", {}) or {}
        for regime, side_map in af.items():
            for side, rho in side_map.items():
                bucket.setdefault((regime, side), {"spearman": []})
                if rho is not None:
                    bucket[(regime, side)]["spearman"].append(float(rho))

    # Global stats (ECE/Brier/fallback) — average across folds
    eces = [float(r["ece"]) for r in fold_metrics if r.get("ece") is not None and np.isfinite(r["ece"])]
    briers = [float(r["brier"]) for r in fold_metrics if r.get("brier") is not None and np.isfinite(r["brier"])]
    fallback_rates = [float(r["fallback_rate"]) for r in fold_metrics if r.get("fallback_rate") is not None and np.isfinite(r["fallback_rate"])]

    mean_ece = float(np.mean(eces)) if eces else float("nan")
    mean_brier = float(np.mean(briers)) if briers else float("nan")
    mean_fallback = float(np.mean(fallback_rates)) if fallback_rates else float("nan")

    # Evaluate gates
    gates = {
        "ece_pass": bool(np.isfinite(mean_ece) and (mean_ece <= thresholds.ece_max)),
        "brier_pass": "UNKNOWN" if dev_brier is None else bool(np.isfinite(mean_brier) and (mean_brier <= (dev_brier + thresholds.brier_additive_worst))),
        "fallback_pass": bool(np.isfinite(mean_fallback) and (mean_fallback <= thresholds.fallback_rate_max)),
        "per_bucket": {}
    }

    for (regime, side), d in bucket.items():
        # If no data, mark UNKNOWN; else check mean Spearman
        if not d["spearman"]:
            gates["per_bucket"][f"{regime}::{side}"] = {"analog_spearman": None, "pass": "UNKNOWN"}
            continue
        s_mean = float(np.mean(d["spearman"]))
        gates["per_bucket"][f"{regime}::{side}"] = {
            "analog_spearman": s_mean,
            "pass": bool(s_mean >= thresholds.analog_spearman_min),
        }

    return {
        "mean_ece": mean_ece,
        "mean_brier": mean_brier,
        "mean_fallback_rate": mean_fallback,
        "gates": gates,
    }

def _portfolio_gate_results(trades: pd.DataFrame, thresholds: GateThresholds) -> dict:
    pm = _portfolio_metrics_from_trades(trades)
    sharpe_ok = np.isfinite(pm["sharpe"]) and (pm["sharpe"] >= thresholds.sharpe_min)
    mdd_ok = np.isfinite(pm["max_drawdown"]) and (abs(pm["max_drawdown"]) <= thresholds.mdd_max_abs)

    slip_med_ok = "UNKNOWN"
    slip_p99_ok = "UNKNOWN"
    if pm.get("slippage_med_abs_bps") is not None:
        slip_med_ok = pm["slippage_med_abs_bps"] <= thresholds.slippage_med_abs_bps_max
    if pm.get("slippage_p99_abs_bps") is not None:
        slip_p99_ok = pm["slippage_p99_abs_bps"] <= thresholds.slippage_p99_abs_bps_max

    return {
        "metrics": pm,
        "gates": {
            "sharpe_pass": bool(sharpe_ok),
            "mdd_pass": bool(mdd_ok),
            "slippage_med_abs_bps_pass": slip_med_ok,
            "slippage_p99_abs_bps_pass": slip_p99_ok,
        }
    }

def _overall_pass(panel: dict) -> bool:
    """All bucket-level gates (where known) + portfolio gates must pass."""
    b = panel["bucket"]["gates"]
    per_bucket = [v["pass"] for v in b["per_bucket"].values()]
    # Treat "UNKNOWN" as neutral (do not fail), only False fails.
    per_bucket_ok = all((x is True or x == "UNKNOWN") for x in per_bucket)
    global_ok = (
        (b["ece_pass"] is True) and
        (b["brier_pass"] is True or b["brier_pass"] == "UNKNOWN") and
        (b["fallback_pass"] is True)
    )
    port = panel["portfolio"]["gates"]
    port_ok = (
        (port["sharpe_pass"] is True) and
        (port["mdd_pass"] is True) and
        (port["slippage_med_abs_bps_pass"] in (True, "UNKNOWN")) and
        (port["slippage_p99_abs_bps_pass"] in (True, "UNKNOWN"))
    )
    return bool(per_bucket_ok and global_ok and port_ok)

def generate_final_report(
    *,
    expanding_root: str | Path,
    rolling_root: str | Path,
    out_dir: str | Path | None = None,
    thresholds: GateThresholds = GateThresholds(),
    dev_brier_reference: float | None = None,
) -> dict:
    """
    Aggregate Expanding vs Rolling panels and evaluate promotion gates.
    Writes: final_report.json and final_report.html
    """
    #exp_root = _resolve_path(expanding_root)
    #rol_root = _resolve_path(rolling_root)

    # Accept either: a dict cfg, or a string/path; normalize through resolver when cfg-like
    if isinstance(expanding_root, dict):
        exp_root = resolve_artifacts_root(expanding_root, create=False)
    else:
        exp_root = Path(expanding_root)

    if isinstance(rolling_root, dict):
        rol_root = resolve_artifacts_root(rolling_root, create=False)
    else:
        rol_root = Path(rolling_root)

    print(f"[RunContext] expanding_artifacts_root={exp_root}")
    print(f"[RunContext] rolling_artifacts_root={rol_root}")

    out = Path(out_dir) if out_dir else exp_root
    out.mkdir(parents=True, exist_ok=True)

    # Load per-fold metrics & trades
    exp_fm = _load_fold_metrics(exp_root)
    rol_fm = _load_fold_metrics(rol_root)
    exp_tr = _concat_trades(exp_root)
    rol_tr = _concat_trades(rol_root)

    # Build panels
    exp_panel = {
        "bucket": _bucket_gate_results(exp_fm, thresholds, dev_brier_reference),
        "portfolio": _portfolio_gate_results(exp_tr, thresholds),
    }
    rol_panel = {
        "bucket": _bucket_gate_results(rol_fm, thresholds, dev_brier_reference),
        "portfolio": _portfolio_gate_results(rol_tr, thresholds),
    }

    # Overall decisions
    exp_pass = _overall_pass(exp_panel)
    # Rolling must not catastrophically contradict Expanding:
    # If Rolling is weaker but within deltas, still pass.
    rol_pass = _overall_pass(rol_panel)
    # Allow "weaker but within deltas" override:
    if not rol_pass and exp_pass:
        # compute rough PF & Sharpe deltas if available
        e_sh = exp_panel["portfolio"]["metrics"].get("sharpe", float("nan"))
        r_sh = rol_panel["portfolio"]["metrics"].get("sharpe", float("nan"))
        # “PF” proxy: net_pnl vs n_trades (naive)
        e_pf = exp_panel["portfolio"]["metrics"].get("net_pnl", 0.0)
        r_pf = rol_panel["portfolio"]["metrics"].get("net_pnl", 0.0)
        sh_ok = (np.isfinite(e_sh) and np.isfinite(r_sh) and (e_sh - r_sh) <= thresholds.delta_sharpe_ok)
        pf_ok = ((e_pf - r_pf) <= thresholds.delta_pf_ok)
        if sh_ok and pf_ok:
            rol_pass = True

    final = {
        "thresholds": asdict(thresholds),
        "expanding": exp_panel,
        "rolling": rol_panel,
        "pass_flags": {
            "expanding": exp_pass,
            "rolling": rol_pass
        },
        "promotion_decision": bool(exp_pass and rol_pass)
    }

    # JSON
    (out / "final_report.json").write_text(json.dumps(final, indent=2), encoding="utf-8")

    # Minimal HTML summary
    html = f"""
    <html><head><title>A2 Promotion Report</title></head>
    <body>
      <h2>A2 Promotion Report</h2>
      <h3>Decision: {"PASS" if final["promotion_decision"] else "FAIL"}</h3>
      <h4>Expanding: {"PASS" if exp_pass else "FAIL"}</h4>
      <pre>{json.dumps(exp_panel, indent=2)}</pre>
      <h4>Rolling: {"PASS" if rol_pass else "FAIL"}</h4>
      <pre>{json.dumps(rol_panel, indent=2)}</pre>
    </body></html>
    """
    (out / "final_report.html").write_text(html, encoding="utf-8")
    return final
