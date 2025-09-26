# scripts/a2_report.py
from __future__ import annotations
import json
from pathlib import Path
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



    artifacts_root = Path(artifacts_root)
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



    # --- NEW: p-spread telemetry on all scanned rows ---
    p_min = float("nan")
    p_max = float("nan")
    p_iqr = float("nan")
    if not dec.empty and "p" in dec.columns and dec["p"].notna().any():
        p_vals = dec["p"].astype(float)
        p_min = float(p_vals.min())
        p_max = float(p_vals.max())
        try:
            q75, q25 = np.percentile(p_vals, [75, 25])
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
        if dec["label"].nunique() > 1:
            auc_all = float(roc_auc_score(dec["label"].to_numpy(), dec["p"].to_numpy()))
    except Exception:
        pass
    try:
        ent = dec[dec["gate"] == 1]
        if len(ent) > 0 and ent["label"].nunique() > 1:
            auc_entries = float(roc_auc_score(ent["label"].to_numpy(), ent["p"].to_numpy()))
    except Exception:
        pass

    # --- Flipped AUCs (sign check) ---
    auc_all_flip = float("nan")
    auc_entries_flip = float("nan")
    try:
        if dec["label"].nunique() > 1:
            p_all = dec["p"].to_numpy(dtype=float)
            y_all = dec["label"].to_numpy(dtype=float)
            auc_all_flip = float(roc_auc_score(y_all, 1.0 - p_all))
    except Exception:
        pass
    try:
        if len(ent) > 0 and ent["label"].nunique() > 1:
            p_ent = ent["p"].to_numpy(dtype=float)
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
    try:
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
        pass

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


