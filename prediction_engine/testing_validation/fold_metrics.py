# prediction_engine/testing_validation/fold_metrics.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class FoldMetricsConfig:
    # column names used in decisions/trades
    p_col: str = "p_cal"                # calibrated probability / score on TEST
    regime_col: str = "regime"          # TREND/RANGE/VOL/GLOBAL
    side_col: str = "side"              # "long"/"short" or +1/-1
    decision_ts_col: str = "timestamp"  # decision timestamp in decisions
    symbol_col: str = "symbol"

    # trades columns (for realized outcome)
    trade_id_col: Optional[str] = "decision_id"  # if present in both frames
    entry_ts_col: str = "entry_ts"
    realized_pnl_col: str = "realized_pnl_after_costs"  # fallback to "realized_pnl" if missing

    # analog columns (optional)
    nn_mu_col_candidates: Tuple[str, ...] = ("nn_mu", "nn_expected_outcome", "mu_hist")

    # fallback flag columns (optional)
    fallback_col_candidates: Tuple[str, ...] = ("fallback", "used_global", "used_default")

    # plotting
    dpi: int = 110


def _resolve_p_col(joined: pd.DataFrame, p_col: str) -> str:
    if p_col in joined.columns:
        return p_col
    # common fallbacks
    for alt in ("p_cal", "p", "prob", "p_pred", "p_hat", "p_raw"):
        if alt in joined.columns:
            return alt
    raise KeyError(
        f"[FoldMetrics] cfg.p_col='{p_col}' not found. "
        f"Available columns include: {list(joined.columns)[:40]} ..."
    )


def _load_decisions_and_trades(decisions_parquet, trades_parquet):
    """
    Robust loader that tolerates schema differences.
    Returns a minimal merged DataFrame with columns:
    ['timestamp', 'p_cal', 'realized_pnl_after_costs'] if available.
    """
    dec = pd.read_parquet(decisions_parquet)
    trd = pd.read_parquet(trades_parquet)

    # Normalize expected column names
    if "timestamp" not in dec.columns:
        for c in ("decision_ts", "ts", "time"):
            if c in dec.columns:
                dec = dec.rename(columns={c: "timestamp"})
                break
    if "entry_ts" not in trd.columns:
        for c in ("timestamp", "ts", "time"):
            if c in trd.columns:
                trd = trd.rename(columns={c: "entry_ts"})
                break

    # Ensure datetime & UTC if present
    if "timestamp" in dec.columns:
        dec["timestamp"] = pd.to_datetime(dec["timestamp"], utc=True, errors="coerce")
    if "entry_ts" in trd.columns:
        trd["entry_ts"] = pd.to_datetime(trd["entry_ts"], utc=True, errors="coerce")

    # Prefer decision_id join if both have it
    if "decision_id" in dec.columns and "decision_id" in trd.columns:
        cols_trd = ["decision_id", "realized_pnl_after_costs"]
        cols_trd = [c for c in cols_trd if c in trd.columns]
        df = dec.merge(trd[cols_trd], on="decision_id", how="inner")

    # Next best: symbol + exact timestamp match
    elif {"symbol", "timestamp"}.issubset(dec.columns) and {"symbol", "entry_ts"}.issubset(trd.columns):
        cols_trd = ["symbol", "entry_ts", "realized_pnl_after_costs"]
        cols_trd = [c for c in cols_trd if c in trd.columns]
        df = dec.merge(
            trd[cols_trd],
            left_on=["symbol", "timestamp"],
            right_on=["symbol", "entry_ts"],
            how="inner",
        )
        if "entry_ts" in df.columns:
            df = df.drop(columns=["entry_ts"])

    # Fallback: align by chronological order
    else:
        dec_sorted = dec.sort_values(dec.columns.intersection(["timestamp"]).tolist() or dec.columns.tolist())
        trd_sorted = trd.sort_values(trd.columns.intersection(["entry_ts", "timestamp"]).tolist() or trd.columns.tolist())
        n = min(len(dec_sorted), len(trd_sorted))
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(
                (dec_sorted["timestamp"].iloc[:n] if "timestamp" in dec_sorted else pd.RangeIndex(n)),
                utc=True, errors="coerce"
            ),
            "p_cal": (dec_sorted["p_cal"].iloc[:n] if "p_cal" in dec_sorted else pd.Series([0.5]*n)),
            "symbol": (dec_sorted["symbol"].iloc[:n] if "symbol" in dec_sorted else pd.Series(["UNK"]*n)),
            "realized_pnl_after_costs": trd_sorted["realized_pnl_after_costs"].iloc[:n].to_numpy()
        })

    # Keep only what we need downstream
    need = ["timestamp", "p_cal", "realized_pnl_after_costs"]
    df = df[[c for c in need if c in df.columns]].copy()
    df = df.sort_values("timestamp") if "timestamp" in df.columns else df
    return df


# -------------------------------
# Core public API
# -------------------------------

def compute_and_save_fold_metrics(
    *,
    decisions_parquet: Path | str,
    trades_parquet: Path | str,
    out_dir: Path | str,
    cfg: FoldMetricsConfig = FoldMetricsConfig(),
    train_feature_samples: Optional[pd.DataFrame] = None,  # optional for PSI
    test_feature_samples: Optional[pd.DataFrame] = None,   # optional for PSI
    psi_features: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Compute per-fold metrics and save:
      - <out_dir>/fold_metrics.json
      - <out_dir>/plots/{reliability.png, decile_lift.png, residual_cusum.png}
    Returns a dict of the main scalar metrics for quick checks.
    """
    out = Path(out_dir)
    (out / "plots").mkdir(parents=True, exist_ok=True)

    # ---- Load frames
    dec_path = Path(decisions_parquet)
    tr_path = Path(trades_parquet)
    if not dec_path.exists():
        raise FileNotFoundError(dec_path)
    if not tr_path.exists():
        # allow empty trades file (no trades): create empty DF
        trades = pd.DataFrame(columns=[cfg.entry_ts_col, cfg.symbol_col, cfg.realized_pnl_col])
    else:
        trades = _safe_read_parquet(tr_path)

    decisions = _safe_read_parquet(dec_path)


    # ---- Ensure probability column exists (prevents KeyError: 'p_cal')
    decisions, _ = _resolve_prob_col(decisions, cfg.p_col)


    if cfg.p_col in decisions.columns:
        # cheap sanity breadcrumb in logs
        print(f"[FoldMetrics] using prob column: {cfg.p_col}")
        print(f"[FoldMetrics] decisions cols sample: {sorted(list(decisions.columns))[:40]}")


    if decisions.empty:
        # write empty outputs and return
        payload = {
            "n_decisions": 0,
            "n_trades_matched": 0,
            "ece": None,
            "brier": None,
            "deciles": 0,
            "fallback_rate": None,
            "analog_fidelity_by_bucket": {},
            "residual_cusum_3sigma_breaches": 0,
            "psi": None,
        }
        (out / "fold_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    # ---- Build realized labels by joining decisions to trades
    joined = _join_decisions_trades(decisions, trades, cfg)

    # realized as win-rate label in {0,1}
    realized = (joined[cfg.realized_pnl_col]
                .fillna(0.0)
                .astype(float) > 0.0).astype(float)
    joined = joined.assign(realized=realized)

    # ---- Calibration metrics (ECE, Brier), decile lift, reliability curve
    cal = _calibration_metrics(joined, cfg)
    _plot_reliability_curve(cal, out / "plots" / "reliability.png", dpi=cfg.dpi)
    _plot_decile_lift(cal, out / "plots" / "decile_lift.png", dpi=cfg.dpi)

    # --- Residual CUSUM plot (realized - p), centered to remove mean drift
    plots_dir = out / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Use the already-joined frame to avoid schema issues
    y = joined["realized"].astype(float).to_numpy()
    #p = joined[cfg.p_col].astype(float).clip(0.0, 1.0).to_numpy()
    p_col = _resolve_p_col(joined, cfg.p_col)
    p = joined[p_col].astype(float).clip(0.0, 1.0).to_numpy()

    resid = y - p
    resid_centered = resid - resid.mean()  # center for CUSUM stability
    cusum = np.cumsum(resid_centered)

    fig, ax = plt.subplots(figsize=(8, 3), dpi=cfg.dpi)
    if cfg.decision_ts_col in joined.columns:
        ax.plot(pd.to_datetime(joined[cfg.decision_ts_col]).to_numpy(), cusum)
        ax.set_xlabel("time")
    else:
        ax.plot(range(len(cusum)), cusum)
        ax.set_xlabel("index")
    ax.set_title("Residual CUSUM (centered realized − p)")
    ax.set_ylabel("cum sum of residuals")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(plots_dir / "residual_cusum.png")
    plt.close(fig)

    # ---- Analog fidelity (Spearman of historical NN μ vs realized), by regime×side
    analog = _analog_fidelity(joined, cfg)

    # ---- Residual QC: CUSUM over residual = realized - p
    residual = joined["realized"] - joined[cfg.p_col].astype(float)
    cusum_breaches = _cusum_breaches(residual)

    # ---- Fallback rate
    fallback_rate = _fallback_rate(decisions, cfg)

    # ---- Optional PSI (feature drift), if samples provided
    psi_payload = None
    if train_feature_samples is not None and test_feature_samples is not None and psi_features:
        psi_payload = _psi(train_feature_samples, test_feature_samples, psi_features)

    # ---- Save JSON payload
    payload = {
        "n_decisions": int(len(decisions)),
        "n_trades_matched": int(joined["__match_flag__"].sum()),
        "ece": float(cal["ece"]) if np.isfinite(cal["ece"]) else None,
        "brier": float(cal["brier"]) if np.isfinite(cal["brier"]) else None,
        "deciles": int(cal["n_deciles"]),
        "decile_table": cal["decile_table"].to_dict(orient="records"),
        "fallback_rate": None if fallback_rate is None else float(fallback_rate),
        "analog_fidelity_by_bucket": analog,
        "residual_cusum_3sigma_breaches": int(cusum_breaches),
        "psi": psi_payload,
        "notes": {
            "ece_bins": "deciles",
            "win_label": f"{cfg.realized_pnl_col} > 0",
            "join_strategy": "decision_id OR (symbol,timestamp)~(symbol,entry_ts)",
        },
    }
    (out / "fold_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


# -------------------------------
# Helpers
# -------------------------------

def _safe_read_parquet(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_parquet(path)


# -------------------------------
# Helpers (add this)
# -------------------------------

def _resolve_prob_col(df: pd.DataFrame, desired: str) -> tuple[pd.DataFrame, str]:
    """
    Ensure a usable probability column exists.

    Returns (df, prob_col_name_in_df).

    Strategy:
      1) If desired exists -> use it.
      2) Else try common aliases -> rename first hit to desired.
      3) Else create desired as NaN (so downstream returns empty metrics instead of crashing).
    """
    if desired in df.columns:
        return df, desired

    # Common aliases seen across pipelines / refactors
    candidates = [
        "p_cal", "p", "prob", "proba", "p_hat", "p_pred", "p_model",
        "p_iso", "p_post", "p_win", "win_prob",
        "score_cal", "score_prob", "score",
    ]

    for c in candidates:
        if c in df.columns:
            # rename the alias to the configured name so the rest of the code stays consistent
            df = df.rename(columns={c: desired})
            return df, desired

    # Nothing found: create a placeholder column to avoid KeyError.
    # Downstream calibration will become empty and return NaNs gracefully.
    df[desired] = np.nan
    return df, desired


def _join_decisions_trades(dec, tr, cfg: FoldMetricsConfig) -> pd.DataFrame:
    dec = dec.copy()
    tr = tr.copy()

    # Ensure prob column exists on decisions before we ever merge
    dec, _ = _resolve_prob_col(dec, cfg.p_col)

    # Ensure realized_pnl column name exists
    if cfg.realized_pnl_col not in tr.columns:
        # fallback alias
        if "realized_pnl" in tr.columns:
            tr = tr.rename(columns={"realized_pnl": cfg.realized_pnl_col})
        else:
            tr[cfg.realized_pnl_col] = np.nan

    # prefer decision_id join if present
    if cfg.trade_id_col and cfg.trade_id_col in dec.columns and cfg.trade_id_col in tr.columns:
        j = dec.merge(
            tr[[cfg.trade_id_col, cfg.symbol_col, cfg.entry_ts_col, cfg.realized_pnl_col]],
            on=cfg.trade_id_col, how="left"
        )
        j["__match_flag__"] = (~j[cfg.realized_pnl_col].isna()).astype(int)
        return j

    # fallback to (symbol, decision_ts) ~ (symbol, entry_ts)
    if cfg.symbol_col in dec.columns and cfg.decision_ts_col in dec.columns \
       and cfg.symbol_col in tr.columns and cfg.entry_ts_col in tr.columns:
        # to_datetime to be safe (keep tz-awareness if present upstream)
        dec_ts = pd.to_datetime(dec[cfg.decision_ts_col], utc=False, errors="coerce")
        tr_ts = pd.to_datetime(tr[cfg.entry_ts_col], utc=False, errors="coerce")
        dec = dec.assign(__j_ts=dec_ts)
        tr = tr.assign(__j_ts=tr_ts)
        j = dec.merge(
            tr[[cfg.symbol_col, "__j_ts", cfg.realized_pnl_col]],
            left_on=[cfg.symbol_col, "__j_ts"],
            right_on=[cfg.symbol_col, "__j_ts"],
            how="left"
        )
        j["__match_flag__"] = (~j[cfg.realized_pnl_col].isna()).astype(int)
        j = j.drop(columns="__j_ts", errors="ignore")
        return j

    # if all else fails, attach empty pnl
    j = dec.copy()
    j[cfg.realized_pnl_col] = np.nan
    j["__match_flag__"] = 0
    return j


'''def _calibration_metrics(joined: pd.DataFrame, cfg: FoldMetricsConfig) -> Dict[str, object]:
    # ---- Robust column selection (prevents KeyError on missing cfg.p_col) ----
    requested = getattr(cfg, "p_col", None) or "p_cal"

    # Try requested first, then common fallbacks used in your pipeline
    candidates = [
        requested,
        "p_cal",
        "p",          # common generic prob
        "p_raw",      # pre-calibration
        "p_hat",
        "p_pred",
        "p_full",     # sometimes used in gating logs
        "p_gate",
        "prob",
        "prob_cal",
    ]

    p_col = next((c for c in candidates if c in joined.columns), None)

    # If we don't have the required columns, skip calibration instead of crashing
    if p_col is None or "realized" not in joined.columns:
        # Return the same shape as your empty-case return
        return {
            "ece": np.nan, "brier": np.nan, "n_deciles": 0,
            "decile_table": pd.DataFrame(columns=["decile", "p_mean", "y_rate", "count"]),
            # Optional: include debug fields; harmless for downstream if ignored
            "p_col_used": p_col,
            "p_col_requested": requested,
        }

    # Ensure prob col exists in joined too (belt + suspenders)
    if cfg.p_col not in joined.columns:
        joined, _ = _resolve_prob_col(joined, cfg.p_col)

    # Build df using the actual available p column, but normalize name to requested
    df = joined[[p_col, "realized"]].dropna().copy()
    if df.empty:
        return {
            "ece": np.nan, "brier": np.nan, "n_deciles": 0,
            "decile_table": pd.DataFrame(columns=["decile", "p_mean", "y_rate", "count"]),
            "p_col_used": p_col,
            "p_col_requested": requested,
        }

    # clip p to [0,1]
    p = df[p_col].astype(float).clip(0.0, 1.0)
    y = df["realized"].astype(float)

    # Brier
    brier = float(np.mean((p - y) ** 2))

    # deciles
    try:
        dec = pd.qcut(p, 10, labels=False, duplicates="drop")
    except ValueError:
        # not enough uniques; single bin
        dec = pd.Series(np.zeros(len(p), dtype=int), index=p.index)

    tab = (
        pd.DataFrame({"p": p, "y": y, "dec": dec})
        .groupby("dec", as_index=False)
        .agg(p_mean=("p", "mean"), y_rate=("y", "mean"), count=("y", "size"))
        .sort_values("dec")
        .rename(columns={"dec": "decile"})
    )
    n = len(df)
    # ECE (expected calibration error) with decile weights
    ece = float(np.sum(np.abs(tab["p_mean"] - tab["y_rate"]) * (tab["count"] / n)))

    return {
        "ece": ece,
        "brier": brier,
        "n_deciles": int(tab.shape[0]),
        "decile_table": tab,
        "p_col_used": p_col,
        "p_col_requested": requested,
    }'''

def _calibration_metrics(joined: pd.DataFrame, cfg: FoldMetricsConfig) -> Dict[str, object]:
    p_col = _resolve_p_col(joined, cfg.p_col)

    df = joined[[p_col, "realized"]].dropna().copy()
    if df.empty:
        return {
            "ece": np.nan, "brier": np.nan, "n_deciles": 0,
            "decile_table": pd.DataFrame(columns=["decile", "p_mean", "y_rate", "count"])
        }

    p = df[p_col].astype(float).clip(0.0, 1.0)
    y = df["realized"].astype(float)

    brier = float(np.mean((p - y) ** 2))

    try:
        dec = pd.qcut(p, 10, labels=False, duplicates="drop")
    except ValueError:
        dec = pd.Series(np.zeros(len(p), dtype=int), index=p.index)

    tab = (
        pd.DataFrame({"p": p, "y": y, "dec": dec})
        .groupby("dec", as_index=False)
        .agg(p_mean=("p", "mean"), y_rate=("y", "mean"), count=("y", "size"))
        .sort_values("dec")
        .rename(columns={"dec": "decile"})
    )

    n = len(df)
    ece = float(np.sum(np.abs(tab["p_mean"] - tab["y_rate"]) * (tab["count"] / n)))
    return {"ece": ece, "brier": brier, "n_deciles": int(tab.shape[0]), "decile_table": tab}



def _plot_reliability_curve(cal: Dict[str, object], out_png: Path, dpi: int = 110) -> None:
    tab: pd.DataFrame = cal["decile_table"]
    if tab.empty:
        # Still produce an empty file to satisfy "file exists" checks
        plt.figure(dpi=dpi)
        plt.title("Reliability curve (no data)")
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        return
    x = tab["p_mean"].to_numpy()
    y = tab["y_rate"].to_numpy()
    plt.figure(dpi=dpi)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.scatter(x, y)
    plt.xlabel("Predicted probability (mean per decile)")
    plt.ylabel("Empirical event rate")
    plt.title("Reliability curve (deciles)")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def _plot_decile_lift(cal: Dict[str, object], out_png: Path, dpi: int = 110) -> None:
    tab: pd.DataFrame = cal["decile_table"]
    if tab.empty:
        # Still produce an empty file to satisfy "file exists" checks
        plt.figure(dpi=dpi)
        plt.title("Decile lift (no data)")
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        return
    plt.figure(dpi=dpi)
    plt.bar(tab["decile"], tab["y_rate"])
    plt.xlabel("Decile (low → high p)")
    plt.ylabel("Empirical event rate")
    plt.title("Decile lift (win-rate by predicted p decile)")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def _analog_fidelity(joined: pd.DataFrame, cfg: FoldMetricsConfig) -> Dict[str, Dict[str, float]]:
    # Find a candidate column with historical NN expectation
    nn_col = None
    for c in cfg.nn_mu_col_candidates:
        if c in joined.columns:
            nn_col = c
            break
    if nn_col is None:
        return {}  # not available → skip

    out: Dict[str, Dict[str, float]] = {}
    # Normalize side to string labels
    side = joined.get(cfg.side_col)
    if side is None:
        side = pd.Series(["long"] * len(joined))
    side = side.map({1: "long", -1: "short"}).fillna(side)

    reg = joined.get(cfg.regime_col, pd.Series(["GLOBAL"] * len(joined)))
    tmp = pd.DataFrame({
        "regime": reg.astype(str),
        "side": side.astype(str),
        "nn_mu": joined[nn_col].astype(float),
        "realized": joined["realized"].astype(float),
    }).dropna()

    # Compute Spearman per bucket
    for (r, s), g in tmp.groupby(["regime", "side"]):
        if len(g) < 8:  # too small for meaningful corr
            continue
        rho = float(g[["nn_mu", "realized"]].corr(method="spearman").iloc[0, 1])
        out.setdefault(r, {})[s] = rho
    return out


def _cusum_breaches(residual: pd.Series) -> int:
    if residual.empty:
        return 0
    r = residual.astype(float)
    cs = r.cumsum()
    s = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    if s == 0.0:
        return 0
    breaches = int((np.abs(cs) > 3.0 * s).sum())
    return breaches


def _fallback_rate(decisions: pd.DataFrame, cfg: FoldMetricsConfig) -> Optional[float]:
    # prefer explicit boolean/flag; else treat regime==GLOBAL as fallback
    for c in cfg.fallback_col_candidates:
        if c in decisions.columns:
            f = decisions[c].astype(bool)
            return float(f.mean())
    if cfg.regime_col in decisions.columns:
        return float((decisions[cfg.regime_col].astype(str) == "GLOBAL").mean())
    return None


# Population Stability Index (optional)
def _psi(train_df: pd.DataFrame, test_df: pd.DataFrame, features: List[str], bins: int = 10) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for col in features:
        a = pd.Series(train_df[col]).astype(float).dropna()
        b = pd.Series(test_df[col]).astype(float).dropna()
        if len(a) < 10 or len(b) < 10:
            out[col] = np.nan
            continue
        qs = np.quantile(a, np.linspace(0, 1, bins + 1))
        qs[0], qs[-1] = -np.inf, np.inf
        a_counts = np.histogram(a, bins=qs)[0].astype(float)
        b_counts = np.histogram(b, bins=qs)[0].astype(float)
        a_prop = np.clip(a_counts / max(a_counts.sum(), 1.0), 1e-6, 1.0)
        b_prop = np.clip(b_counts / max(b_counts.sum(), 1.0), 1e-6, 1.0)
        psi = float(np.sum((a_prop - b_prop) * np.log(a_prop / b_prop)))
        out[col] = psi
    return out
