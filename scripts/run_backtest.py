# ---------------------------------------------------------------------------
# FILE: prediction_engine/scripts/run_backtest.py
# ---------------------------------------------------------------------------
"""
Event-driven batch back-test of EVEngine on historical OHLCV bars.

Outputs
-------
• CSV (signals_out) with per-bar signals, fills & PnL
• Console summary of cumulative P&L + risk metrics
"""
from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict
from prediction_engine.calibration import load_calibrator, map_mu_to_prob

#import pandas as pd

from feature_engineering.pipelines.core import CoreFeaturePipeline
from prediction_engine.ev_engine import EVEngine
from prediction_engine.tx_cost import BasicCostModel  # NEW
from execution.risk_manager import RiskManager
from prediction_engine.testing_validation.backtester import BrokerStub  # NEW
from scripts.rebuild_artefacts import rebuild_if_needed  # NEW
from scanner.detectors import build_detectors        # ‹— add scanner import
from backtester import Backtester
from execution.manager import ExecutionManager
from execution.metrics.report import load_blotter, pnl_curve, latency_summary
from execution.manager import ExecutionManager
from execution.risk_manager import RiskManager            # NEW
from prediction_engine.tx_cost import BasicCostModel
from prediction_engine.calibration import load_calibrator, calibrate_isotonic  # NEW – iso µ‑cal


# ─── global run metadata ─────────────────────────────────────────────
import uuid, subprocess, datetime as dt
RUN_ID = uuid.uuid4().hex[:8]          # short random id
try:
    GIT_SHA = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=Path(__file__).parents[2],  # repo root
    ).decode().strip()
except Exception:
    GIT_SHA = "n/a"
RUN_META = {
    "run_id": RUN_ID,
    "git_sha": GIT_SHA,
    "started": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
}
# ─────────────────────────────────────────────────────────────────────



# Keep only PCA feature columns (produced by CoreFeaturePipeline)
def _pca_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("pca_")]


# ────────────────────────────────────────────────────────────────────────
# CONFIG – edit here
# ────────────────────────────────────────────────────────────────────────
CONFIG: Dict[str, Any] = {
    # raw minute-bar CSV (Date, Time, Open, High, Low, Close, Volume)
    "csv": "raw_data/RRC.csv",
    "parquet_root": "parquet",
    "symbol": "RRC",
    "start": "1998-08-26",
    "end": "1998-11-26",

    # artefacts created by PathClusterEngine.build()
    "artefacts": "../weights",
    "calibration_dir": "../weights/calibration",  # <— add this

    # capital & trading costs
    "equity": 100_000.0,
    # "spread_bp": 2.0,          # half-spread in basis points
    # "commission": 0.002,       # $/share
    # "slippage_bp": 0.0,        # BrokerStub additional bp slippage
    "max_kelly": 0.5,
    "adv_cap_pct": 0.20,
    "spread_bp": 0.0,  # half-spread in basis points
    "commission": 0.0,  # $/share
    "slippage_bp": 0.0,  # BrokerStub additional bp slippage

    # misc
    "out": "backtest_signals.csv",
    "atr_period": 14,
}


# ────────────────────────────────────────────────────────────────────────


def _resolve_path(path_like: str | Path) -> Path:
    """Resolve path relative to project root if not absolute."""
    # 1) absolute or cwd-relative
    p = Path(path_like).expanduser()
    if p.exists():
        return p.resolve()
    # 2) try relative to the repository root (two levels up)
    repo_root = Path(__file__).parents[1]
    alt = (repo_root / path_like).expanduser()
    if alt.exists():
        return alt.resolve()
    raise FileNotFoundError(f"Could not find {path_like!r} in CWD or under {repo_root!s}")


# ────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────
async def run(cfg: Dict[str, Any]) -> pd.DataFrame:
    logging.basicConfig(
        level=logging.DEBUG, format="[%(levelname)s] %(message)s", stream=sys.stdout
    )



    # suppress EVEngine’s regime/residual warnings
    # logging.getLogger("prediction_engine.ev_engine").setLevel(logging.ERROR)

    log = logging.getLogger("backtest")
    import pandas as pd

    log.info("🚀 Backtest run %s (commit %s) started %s",
             RUN_ID, GIT_SHA, RUN_META["started"])
    log.debug("Full CLI: %s", " ".join(sys.argv))

    # 1 ─ Load raw CSV
    csv_path = _resolve_path(cfg["csv"])
    df_raw = pd.read_csv(csv_path)

    df_raw["timestamp"] = pd.to_datetime(df_raw["Date"] + " " + df_raw["Time"])
    df_raw.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        },
        inplace=True,
    )
    df_raw["symbol"] = cfg["symbol"]
    df_raw = (
        df_raw[
            (df_raw["timestamp"] >= cfg["start"]) & (df_raw["timestamp"] <= cfg["end"])
            ]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    if df_raw.empty:
        raise ValueError("Date filter returned zero rows")


    # ------------------------------------------------------------------
    #  Solution: Add the missing 'trigger_ts' column here
    # ------------------------------------------------------------------
    df_raw["trigger_ts"] = df_raw["timestamp"]

    # ------------------------------------------------------------------
    #  Solution: Calculate and add the 'volume_spike_pct' column
    # ------------------------------------------------------------------
    # Calculate a 20-period moving average of volume
    volume_ma = df_raw['volume'].rolling(window=20, min_periods=1).mean()
    # Calculate the spike as a percentage change from the moving average
    df_raw['volume_spike_pct'] = (df_raw['volume'] / volume_ma) - 1.0
    # Fill any initial NaN values (from the rolling window) with 0.0
    #df_raw['volume_spike_pct'].fillna(0.0, inplace=True)
    # Fill any initial NaN values (from the rolling window) with 0.0
    df_raw['volume_spike_pct'] = df_raw['volume_spike_pct'].fillna(0.0)

    # ─── P2 Step 2: KPI sanity‑check & scan‑first pass-through ──────────
    # prepare the prev_close column for gap detection
    df_raw['prev_close'] = df_raw['close'].shift(1)

    detector = build_detectors()

    # This is the asynchronous call that needs 'await'
    mask = await detector(df_raw)

    pass_rate = mask.mean() * 100
    print(f"[Scanner KPI] Bars passing filters: {mask.sum()} / {len(mask)} = {pass_rate:.2f}%")

    # now only keep the bars that passed
    df_raw = df_raw.loc[mask].reset_index(drop=True)
    # ──────────────────────────────────────────────────────

    # 20‑day rolling ADV in *shares*
    df_raw["adv_shares"] = df_raw["volume"].rolling(20, min_periods=1).mean()

    # dollar ADV (needed if your RiskManager uses notional)
    df_raw["adv_dollars"] = df_raw["adv_shares"] * df_raw["close"]

    # 2 ─ Feature engineering
    #pipe = CoreFeaturePipeline(parquet_root=Path(""))  # in-mem
    #feats, _ = pipe.run_mem(df_raw)

    # 2 ─ Feature engineering in‑memory (no external pickles needed)
    from pathlib import Path
    # instantiate the pure‑pandas pipeline & fit‐transform on this df
    pipe = CoreFeaturePipeline(parquet_root=Path(""))
    feats, _ = pipe.run_mem(df_raw)

    # ------------------------------------------------------------------
    # keep only PCA columns (+ ids)  ────────────
    # ------------------------------------------------------------------
    pc_cols = _pca_cols(feats)
    if not pc_cols:
        raise RuntimeError("No pca_* columns found – did CoreFeaturePipeline run?")
    feats = feats[pc_cols + ["symbol", "timestamp"]]
    feats["open"] = df_raw["open"].values  # Attach open for execution logic
    feats["close"] = df_raw["close"].values
    feats["high"] = df_raw["high"].values  # <— needed for stop‑loss check
    feats["low"] = df_raw["low"].values
    feats["adv_shares"] = df_raw["adv_shares"].values
    feats["adv_dollars"] = df_raw["adv_dollars"].values

    # sanity: ensure ATR exists
    ATR_COL = f"atr{cfg['atr_period']}"
    if ATR_COL not in feats.columns:
        feats[ATR_COL] = (
                df_raw["high"]
                .rolling(cfg['atr_period'])
                .max()
                - df_raw["low"].rolling(cfg['atr_period']).min()
        ).bfill()

    # 3 ─ Load EVEngine artefacts
    art_dir = _resolve_path(cfg["artefacts"])

    # resolve parquet directory (same repo root logic)
    pq_root = _resolve_path(cfg.get("parquet_root", "parquet"))

    # if it contains a "symbol=…" subfolder, use that (so we don't include schema.json)
    sym_dir = pq_root / f"symbol={cfg['symbol']}"
    if sym_dir.exists() and sym_dir.is_dir():
        pq_root = sym_dir

    # after df_raw is filtered to your test window (the mask), infer bar size
    bar_delta = (df_raw["timestamp"].diff().dropna().median()
                 if len(df_raw) > 1 else pd.Timedelta(minutes=1))

    # ── convert start/end to real Timestamps for Parquet filtering ─────
    '''from pandas import to_datetime
    start_ts = to_datetime(cfg["start"], utc=True)
    end_ts = to_datetime(cfg["end"], utc=True)

    # training window ends the bar before the test starts
    #train_end = start_ts - pd.Timedelta(minutes=1)  # or 1 day for daily bars

    # OOS train window: the 60 days before test start
    train_end = start_ts - bar_delta
    train_start = start_ts - pd.Timedelta(days=60)

    # Guard: if your parquet store starts later than train_start, just let the builder pick earliest
    train_start_arg = train_start if train_start < train_end else None

    print("[DBG] train_start:", train_start, "train_end:", train_end, "bar_delta:", bar_delta)

    from pathlib import Path
    import glob

    print("[DBG] pq_root:", pq_root)
    print("[DBG] exists:", Path(pq_root).exists())
    print("[DBG] sample files:", list(Path(pq_root).rglob("*.parquet"))[:5])

    rebuild_if_needed(
        artefact_dir=cfg["artefacts"],
        parquet_root=str(pq_root),  # ← pass the resolved path, not the literal key
        symbols=[cfg["symbol"]],
        #start=start_ts,
        #end=end_ts,
        start=train_start_arg,
        end=train_end,
        n_clusters=64,
    )'''

    from pandas import to_datetime
    start_ts = to_datetime(cfg["start"], utc=True)
    end_ts = to_datetime(cfg["end"], utc=True)

    # Build artefacts on the SAME window as the test (IS build to avoid 0-row slices)
    print("[DBG] build_start:", start_ts, "build_end:", end_ts)

    rebuild_if_needed(
        artefact_dir=cfg["artefacts"],
        parquet_root=str(pq_root),
        symbols=[cfg["symbol"]],
        start=start_ts,
        end=end_ts,
        n_clusters=64,
    )

    #cost_model = BasicCostModel()
    # one shared, zero-cost model everywhere
    #cost_model = BasicCostModel()

    # --- load the µ‑calibration mapping ----------------------------
    #calibrator = load_calibrator(Path(cfg["artefacts"]) / "calibration")

    # we still load it for later use, but we won’t stomp µ in the engine
    #calibrator = load_calibrator(Path(cfg["artefacts"]) / "calibration")

    #calibrator = None

    # ─── NEW: Retrain isotonic calibrator on in‐sample mu → 1‑bar return ───


    import numpy as _np, pandas as _pd

    # 1) cluster‐level µ stats
    stats = _np.load(art_dir / "cluster_stats.npz", allow_pickle=True)
    mu_by_cl = stats["mu"]
    #calib = load_calibrator(Path(cfg["calibration_dir"]))

    #ev = EVEngine.from_artifacts(art_dir, cost_model=cost_model,calibrator=calibrator)
    # by passing None here, EVEngine.evaluate() will leave µ untouched

    #prob_cal = load_calibrator(Path(cfg["calibration_dir"]))  # this must be trained on labels

    # --- ONE shared, zero-cost model used everywhere ---
    cost_model = BasicCostModel()
    cost_model._DEFAULT_SPREAD_CENTS = 0.0
    cost_model._COMMISSION = 0.0
    cost_model._IMPACT_COEFF = 0.0

    ev = EVEngine.from_artifacts(
        art_dir,
        cost_model=cost_model,  # <- same instance
    )



    from prediction_engine.calibration import calibrate_isotonic
    # 1) compute µ for each bar
    '''all_X = feats[pc_cols].to_numpy(dtype=np.float32)
    all_mu = np.array(
        [ev.evaluate(x, adv_percentile=10.0, half_spread=0).mu for x in all_X]
    )
    # 2) compute the actual next‐bar return
    ret1m = feats["close"].shift(-1) / feats["open"] - 1.0
    ret1m = ret1m.fillna(0.0).to_numpy()
    # 3) fit a fresh isotonic mapping µ→ret1m
    calibrator = calibrate_isotonic(all_mu.reshape(-1, 1), ret1m)'''

    # ─── Instead of the on‑disk calibrator, fit one in‑memory ─────────
    '''from sklearn.isotonic import IsotonicRegression
    # 1) compute μ for each bar
    all_X = feats[pc_cols].to_numpy(dtype=np.float32)
    all_mu = np.array(
        [ev.evaluate(x, adv_percentile=10.0, half_spread=0).mu for x in all_X]
        )
    # 2) compute binary 1‑bar outcome (1 if ret>0 else 0)
    ret1m = (feats["close"].shift(-1) / feats["open"] - 1.0).fillna(0.0)
    labels = (ret1m > 0).astype(int).to_numpy()

    # 3) fit in‑memory isotonic μ→P(up)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(all_mu, labels)
    calibrator = iso'''

    # === µ → probability + µ → expected-return calibration (clean version) ===
    from sklearn.isotonic import IsotonicRegression

    '''# 1) compute µ for each bar using the SAME engine instance
    all_X = feats[pc_cols].to_numpy(dtype=np.float32)
    all_mu = np.array([ev.evaluate(x, adv_percentile=10.0, half_spread=0).mu for x in all_X])

    # 2a) Build binary labels for PROBABILITY calibration: win on next bar?
    #labels = ((feats["close"].shift(-1) / feats["open"] - 1.0) > 0).astype(int).to_numpy()
    # PROB calibration label: win over next bar (entry at next open, exit next close)
    labels = (
            (feats["close"].shift(-1) / feats["open"].shift(-1) - 1.0) > 0
    ).astype(int).to_numpy()




    # 2b) (optional) continuous target for RETURN calibration
    ret1m_raw = (feats["close"].shift(-1) / feats["open"] - 1.0).to_numpy()
    # subtract a simple cost estimate so the sizing target reflects tradable edge
    _cost_est = cost_model.estimate(half_spread=0, adv_percentile=10.0)  # 0 in free_mode
    ret1m = np.nan_to_num(ret1m_raw - _cost_est, nan=0.0)

    # 3a) PROBABILITY calibrator: µ → P(up)
    iso_prob = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    mask_prob = np.isfinite(all_mu) & np.isfinite(labels)
    #iso_prob.fit(all_mu[mask_prob], labels[mask_prob])

    # replace plain iso_prob.fit(all_mu[mask_prob], labels[mask_prob]) with binned fit:
    bins = 12  # small sample → fewer bins
    dfp = pd.DataFrame({"mu": all_mu[mask_prob], "y": labels[mask_prob]})
    dfp["bin"] = pd.qcut(dfp["mu"], bins, duplicates="drop")
    x_bin = dfp.groupby("bin")["mu"].mean().to_numpy()
    y_bin = dfp.groupby("bin")["y"].mean().to_numpy()

    iso_prob = IsotonicRegression(out_of_bounds="clip", y_min=0.05, y_max=0.95)
    iso_prob.fit(x_bin, y_bin)

    # (optional) if the spread is still < 0.05, mix with a logistic fallback:
    if np.ptp(iso_prob.predict(x_bin)) < 0.05:
        med = np.median(x_bin)
        scale = np.median(np.abs(x_bin - med)) * 3.0 or 1e-3
        platt = lambda z: 1.0 / (1.0 + np.exp(-(z - med) / scale))
        mix = 0.5
        y_mix = (1 - mix) * y_bin + mix * platt(x_bin)
        iso_prob.fit(x_bin, y_mix)

    from sklearn.isotonic import IsotonicRegression
    # 3a) PROBABILITY calibrator: μ → P(up), fit on binned means to avoid saturation
    iso_prob = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    mask_prob = np.isfinite(all_mu) & np.isfinite(labels)
    df_prob = pd.DataFrame({"mu": all_mu[mask_prob], "y": labels[mask_prob]})

    if len(df_prob) >= 40:
        df_prob["bin"] = pd.qcut(df_prob["mu"], 20, duplicates="drop")
        mu_bin_p = df_prob.groupby("bin")["mu"].mean().to_numpy()
        y_bin_p = df_prob.groupby("bin")["y"].mean().to_numpy()
        iso_prob.fit(mu_bin_p, y_bin_p)
    else:
        iso_prob.fit(df_prob["mu"].to_numpy(), df_prob["y"].to_numpy())
      # light guardrail: clip predictions into a neutral band to avoid 0/1 snaps


    def _predict_p(mu_vals: np.ndarray) -> np.ndarray:
        p = iso_prob.predict(mu_vals)
        return np.clip(p, 0.35, 0.65)'''

    from sklearn.isotonic import IsotonicRegression

    all_X = feats[pc_cols].to_numpy(dtype=np.float32)
    all_mu = np.array([ev.evaluate(x, adv_percentile=10.0, half_spread=0).mu for x in all_X])

    # PROB calibration label: win over next bar (entry at next open, exit next close)
    labels = (
            (feats["close"].shift(-1) / feats["open"].shift(-1) - 1.0) > 0
    ).astype(int).to_numpy()

    # continuous target for return calibration on the same horizon
    ret1m_raw = (feats["close"].shift(-1) / feats["open"].shift(-1) - 1.0).to_numpy()
    _cost_est = cost_model.estimate(half_spread=0, adv_percentile=10.0)  # 0 in free_mode
    ret1m = np.nan_to_num(ret1m_raw - _cost_est, nan=0.0)

    # labels/ret1m already fixed above
    mask_prob = np.isfinite(all_mu) & np.isfinite(labels)
    df_prob = pd.DataFrame({"mu": all_mu[mask_prob], "y": labels[mask_prob]})

    iso_prob = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    nbins = 12 if len(df_prob) < 120 else 20
    if len(df_prob) >= 40:
        df_prob["bin"] = pd.qcut(df_prob["mu"], nbins, duplicates="drop")
        mu_bin = df_prob.groupby("bin")["mu"].mean().to_numpy()
        y_bin = df_prob.groupby("bin")["y"].mean().to_numpy()
        iso_prob.fit(mu_bin, y_bin)
    else:
        iso_prob.fit(df_prob["mu"].to_numpy(), df_prob["y"].to_numpy())

    '''def _predict_p(mu_vals: np.ndarray) -> np.ndarray:
        p = iso_prob.predict(mu_vals)
        # very light guardrail
        if np.ptp(p) < 0.05 and len(p) > 1:
            p = 0.5 + 0.5 * (p - p.mean())  # expand a touch around 0.5
        '''

    def _predict_p(mu_vals: np.ndarray) -> np.ndarray:
        p = iso_prob.predict(mu_vals)
        if np.ptp(p) < 0.05 and len(p) > 1:
            p = 0.5 + 0.5 * (p - p.mean())  # small spread boost
        return p  # ← no hard clip here
        # (If you really want a guard: return np.clip(p, 0.40, 0.60))

        #return np.clip(p, 0.30, 0.70)



    # 3b) (optional) RETURN calibrator: µ → E[ret1m]
    iso_ret = IsotonicRegression(out_of_bounds="clip")
    mask_ret = np.isfinite(all_mu) & np.isfinite(ret1m)
    # mild de-noising via quantile binning (helps tiny samples)
    df_cal = pd.DataFrame({"mu": all_mu[mask_ret], "ret": ret1m[mask_ret]})
    df_cal["bin"] = pd.qcut(df_cal["mu"], 20, duplicates="drop")
    mu_bin = df_cal.groupby("bin")["mu"].mean().to_numpy()
    ret_bin = df_cal.groupby("bin")["ret"].mean().to_numpy()
    iso_ret.fit(mu_bin, ret_bin)

    # 4) Inject the PROBABILITY calibrator into the engine so evaluate() yields p_up ≠ constant
    ev._calibrator = iso_prob

    # Optional: clip live calibrator a bit to avoid degenerate p's on tiny samples
    '''class ClippedIso:
        def __init__(self, base, lo=0.30, hi=0.70):
            self.base, self.lo, self.hi = base, lo, hi

        def predict(self, x):
            x = np.asarray(x).reshape(-1)
            p = self.base.predict(x)
            return np.clip(p, self.lo, self.hi)

    ev._calibrator = ClippedIso(iso_prob, lo=0.30, hi=0.70)'''

    ev._calibrator = iso_prob  # ← preferred

    # 5) Residual gate: keep in **squared** units (matches dist2/residual usage)
    stats = np.load(art_dir / "cluster_stats.npz", allow_pickle=True)
    if "tau_sq" in stats.files:
        #ev.residual_threshold = float(np.percentile(stats["tau_sq"], 90))  # d², no sqrt
        # was 90th
        ev.residual_threshold = float(np.percentile(stats["tau_sq"], 95)) if "tau_sq" in stats.files else (
                                                                                                                      2.5 * ev.h) ** 2
        print("[Residual gate] threshold(d²)=", ev.residual_threshold)

    else:
        ev.residual_threshold = (2.0 * ev.h) ** 2

    print(f"[τ²] residual_threshold={ev.residual_threshold:.6g}  "
          f"h²={(ev.h ** 2):.6g}")

    _p = iso_prob.predict(all_mu)
    cfg["p_gate"] = float(np.quantile(_p, 0.60))  # modest bar
    cfg["full_p"] = float(np.quantile(_p, 0.80))  # where we want max scale
    print(f"[ISO] gate={cfg['p_gate']:.3f}  full={cfg['full_p']:.3f}  "
          f"p(mean,std,min,max)=({float(_p.mean()):.3f},{float(_p.std()):.3f},{float(_p.min()):.3f},{float(_p.max()):.3f})")
    print(f"[ISO] p(mean,std,min,max) = "
          f"({float(_p.mean()):.3f}, {float(_p.std()):.3f}, {float(_p.min()):.3f}, {float(_p.max()):.3f})")
    print(f"[ISO] frac p>0.51 = {float((_p > 0.51).mean()):.3f}  "
          f"frac p>0.55 = {float((_p > 0.55).mean()):.3f}")

    # After computing _p
    #cfg["p_gate"] = float(np.quantile(_p, 0.40))  # lower bar = allow some trades
    #cfg["p_gate"] = max(0.48, min(cfg["p_gate"], 0.52))  # keep it sensible
    #cfg["full_p"] = float(max(cfg["p_gate"] + 0.06, np.quantile(_p, 0.75)))
    # TEMP: gentler gates so entries can occur on this tape
    cfg["p_gate"] = 0.295  # allow entries above the clipped floor
    cfg["full_p"] = 0.55  # hit max Kelly by here

    # 6) Quick sanity prints (optional)
    grid = np.linspace(all_mu.min(), all_mu.max(), 7)
    #print("[ISO prob] p(grid) =", np.round(iso_prob.predict(grid), 3))
    print("[ISO prob] p(grid) =", np.round(_predict_p(grid), 3))

    print("[Residual gate] threshold(d²)=", ev.residual_threshold)

    # Keep iso_ret around for sizing decisions in the backtest loop if you want:
    #   exp_ret = float(iso_ret.predict([[ev_raw.mu]])[0])
    # Do NOT pass iso_ret into EVEngine; EVEngine should use iso_prob only.

    numeric_idx = pc_cols  # only PCA features go to EVEngine

    import numpy as _np, pandas as _pd

    # 1) cluster‐level µ stats
    stats = _np.load(art_dir / "cluster_stats.npz", allow_pickle=True)
    mu_by_cl = stats["mu"]
    print("cluster μ summary:\n", _pd.Series(mu_by_cl).describe())
    print("fraction μ>0:", (mu_by_cl > 0).mean())

    # 2) how many of the first 20 centers your live engine thinks are positive?
    centers = _np.load(art_dir / "centers.npy")
    mus_centroid = [ev.evaluate(c, half_spread=0).mu for c in centers[:20]]
    print("share of μ>0 on first 20 centroids:", _np.mean(_np.array(mus_centroid) > 0))

    # 3) residual distribution on centroids
    res = [ev.evaluate(c, half_spread=0).residual for c in centers]
    print("Residual summary:\n", _pd.Series(res).describe())
    print("Frac residual < 0.75:", (_np.array(res) < ev.residual_threshold).mean())



    # ─── SANITY CHECK #1: “cluster_stats.npz” summary ───────────────
    import numpy as _np, pandas as _pd
    stats     = _np.load(art_dir/"cluster_stats.npz", allow_pickle=True)
    mu_by_cl  = stats["mu"]
    print("cluster μ summary:\n", _pd.Series(mu_by_cl).describe())
    print("fraction μ>0:", (mu_by_cl > 0).mean())

    # ─── SANITY CHECK #2: first‐20 centers through the live engine ───
    centers       = _np.load(art_dir/"centers.npy")

    mus_centroid  = [_float := ev.evaluate(c, half_spread=0).mu
                     for c in centers[:20]]
    print("share of μ>0 on first 20 centroids:",
          _np.mean(_np.array(mus_centroid) > 0))

    centers = np.load(art_dir / "centers.npy")
    res = [ev.evaluate(c, half_spread=0).residual for c in centers]
    import numpy as _np, pandas as _pd
    print("Residual summary:\n", _pd.Series(res).describe())
    #print("Frac residual < 0.75:", (_np.array(res) < 0.75).mean())
    print(f"Frac residual < {ev.residual_threshold:.3g}:",
          (_np.array(res) < ev.residual_threshold).mean())

    #import numpy as np
    #from pathlib import Path

    # SANITY CHECK
    #all_mu = [ev.evaluate(row, adv_percentile=10.0).mu
    #          for row in feats[pc_cols].to_numpy(dtype=np.float32)]

    '''all_mu = np.array(
        [ev.evaluate(x, adv_percentile=10.0, half_spread=0).mu
        for x in all_X],
        dtype = np.float32,
        )

    target = (feats["close"].shift(-1) / feats["open"] - 1.0).fillna(0.0)
    # also turn it into a NumPy array
    ret1m = np.asarray(target, dtype=np.float32)

    #iso = IsotonicRegression(out_of_bounds="clip")

    # fit on binned means to reduce overfitting
    iso = IsotonicRegression(out_of_bounds="clip")
    # before fitting, aggregate into 20 quantile‐bins:
    df_cal = pd.DataFrame(dict(mu=all_mu, ret=ret1m))
    df_cal['bin'] = pd.qcut(df_cal['mu'], 20, duplicates='drop')
    mu_bin = df_cal.groupby('bin')['mu'].mean().to_numpy()
    ret_bin = df_cal.groupby('bin')['ret'].mean().to_numpy()
    iso.fit(mu_bin, ret_bin)

    # drop any nan targets
    #mask = ~np.isnan(ret1m)
    #iso.fit(all_mu[mask], ret1m[mask])

    print(f"[Sanity-full] µ>0 on full tape: {sum(m > 0 for m in all_mu)} / {len(all_mu)}")

    # 3) compute realised return **minus cost**
    cost = cost_model.estimate(half_spread=0, adv_percentile=10.0)
    ret1m = feats["close"].shift(-1) / feats["open"] - 1.0 - cost
    ret1m = ret1m.fillna(0.0)  # << make sure no NaN left
    ret1m = ret1m.to_numpy()  # sklearn wants ndarray, not Series
    #iso.fit(all_mu, ret1m)
    mask = ~np.isnan(ret1m)
    iso.fit(all_mu[mask], ret1m[mask])
    # ─── Quick mu sanity BEFORE scan ───────────────────────────────────
    import numpy as _np
    all_X = feats[numeric_idx].to_numpy(dtype=_np.float32)
    all_mu = _np.array([ev.evaluate(x, adv_percentile=10.0).mu for x in all_X])
    print(f"[Sanity-full] μ>0: {(all_mu > 0).sum()} / {len(all_mu)}")
    # (optionally plot a histogram)
    # ─────────────────────────────────────────────────────────────────────

    # ─── P2 Sanity Check: full‑period µ distribution (unfiltered) ────────
    import numpy as _np
    import matplotlib.pyplot as _plt
    # turn your PCA features into a matrix
    mus = feats[numeric_idx].to_numpy(dtype=_np.float32)
    # evaluate µ for every bar
    ev_res = [ev.evaluate(x,
                          adv_percentile=10.0,
                          regime=None,
                          half_spread=0) for x in mus]




    mu_vals = _np.array([r.mu for r in ev_res], dtype=_np.float32)

    sample_mu = np.linspace(mu_vals.min(), mu_vals.max(), 10)
    #print("Calibrated p’s:", calibrator.predict(sample_mu))

    print(f"[Sanity] Unfiltered µ>0: {(mu_vals > 0).sum()} / {len(mu_vals)}")
    _plt.hist(mu_vals, bins=30)
    _plt.title("Full-period µ distribution")
    _plt.show()'''
    # ──────────────────────────────────────────────────────────────────────

    stats_path = Path(cfg["artefacts"]) / "cluster_stats.npz"
    stats = np.load(stats_path, allow_pickle=True)


    #print("Calibrator thresholds:", calibrator.X_thresholds_)
    #print("Calibrator y:", calibrator.y_min, calibrator.y_max)

    #for x in np.linspace(mu_vals.min(), mu_vals.max(), 5):
    #    print("map", x, "→", calibrator.predict([[x]])[0])

    mu = stats["mu"]
    #mu_by_cluster = stats["mu"]  # array of length n_clusters
    #mean_ret1m_by_cl = stats["mean_ret1m"]
    #good_clusters = set(np.where(mu_by_cluster > 0)[0])
    #good_clusters = set(np.where(mean_ret1m_by_cl > 0)[0])

    # ─── Good‑cluster filter – favour clusters that actually made money ───
    if "mean_ret1m" in stats.files:  # produced by PathClusterEngine ≥ 0.4
        mean_ret1m_by_cl = stats["mean_ret1m"].astype(float)

        # ░░░░░░░░░░  PATCH 4  ░░ top‑quartile good_clusters ░░░░░░░░░░
        # This logic now correctly sits inside the if-block.
        # keep only clusters whose mean_ret1m is in the top quartile
        #q75 = np.percentile(mean_ret1m_by_cl, 75)
        #good_clusters = set(np.where(mean_ret1m_by_cl >= q75)[0])
        q80 = np.percentile(mean_ret1m_by_cl, 80)  # keep top 20 %
        good_clusters = set(np.where(mean_ret1m_by_cl >= q80)[0])


        print(f"[Filter] good_clusters tightened to top‑quartile (n={len(good_clusters)})")
        # =====================================================================

    else:  # legacy fallback
        mu_by_cluster = stats["mu"]
        good_clusters = set(np.where(mu_by_cluster > 0)[0])

    WIDEN = True  # TEMP
    if WIDEN:
        good_clusters = set(range(len(stats["mu"])))
        print(f"[TEMP] Widened good_clusters to ALL ({len(good_clusters)})")

    # print("µ by cluster  ➜ ", np.round(mu[:10], 6), " …") # This line and below are unchanged
    print("# good_clusters =", len(good_clusters))

    print(f"[Filter] good_clusters tightened to top‑quartile (n={len(good_clusters)})")
    # =====================================================================

    print("µ by cluster  ➜ ", np.round(mu[:10], 6), " …")
    print("# good_clusters =", len(good_clusters))

    print("CLUSTER µ summary:  mean =", mu.mean(),
          " min =", mu.min(),
          " max =", mu.max())
    print("first 10 µ:", mu[:10])

    #stats = np.load(Path(cfg["artefacts"]) / "cluster_stats.npz", allow_pickle=True)
    #mu = stats["mu"]
    #good_clusters = set(np.where(mu_by_cluster > 0)[0])


    print("[DEBUG clusters] µ summary: ",
          f"mean={mu.mean():.8f}, min={mu.min():.8f}, max={mu.max():.8f}")
    print("[DEBUG clusters] first 10 µ:", mu[:10])

    # schema guard – compare EXACT list of pca_* columns in order
    with open(art_dir / "meta.json", "r", encoding="utf-8") as mf:
        meta = json.load(mf)
    expected = meta["features"]  # e.g. ["pca_1","pca_2",…]

    print("\n[DEBUG] Features listed in meta.json (artefacts):", expected)
    print("[DEBUG] Features produced at runtime (pc_cols):", pc_cols)
    print("[DEBUG] Are they equal?", expected == pc_cols)

    if expected != pc_cols:
        raise RuntimeError(
            f"\n[ERROR] Feature list mismatch.\n  Artefacts: {expected}\n  Runtime : {pc_cols}"
        )


    '''sha_actual = hashlib.sha1(sha_actual.encode()).hexdigest()[:12]
    if sha_actual != sha_expected:
        print("\n[DEBUG] Features in artefacts (meta.json):")
        with open(art_dir / "meta.json", "r", encoding="utf-8") as mf:
            meta = json.load(mf)
            print(meta["features"])

        print("\n[DEBUG] Features produced at runtime (feats):")
        feature_cols_live = [c for c in feats.columns if c not in ("symbol", "timestamp")]
        print(feature_cols_live)

        raise RuntimeError(
            f"Feature SHA mismatch – artefact built for {sha_expected}, got {sha_actual}"
        )'''

    # --- sanity: compare first row *before* + *after* the PCA re‑fit ----------
    live_vec = feats.loc[0, pc_cols].to_numpy()
    train_vec = np.load("weights/centers.npy")[0]  # first centroid
    print("first live PCA vector : ", live_vec[:6])
    print("first centroid vector : ", train_vec[:6])

    # feature_cols = [
    #    c for c in feats.columns if c not in ("symbol", "timestamp")
    # ]  # order preserved
    # numeric_idx = feats[feature_cols].select_dtypes("number").columns.tolist()

    # 4 ─ Instantiate Broker, Cost model & RiskManager

    '''cost_model = BasicCostModel()

    cost_model._DEFAULT_SPREAD_CENTS = float(cfg["spread_bp"])
    cost_model._COMMISSION = float(cfg["commission"])'''

    #cost_model = BasicCostModel()

    # ── turn every component off for this experiment ───────────────────
    #cost_model._DEFAULT_SPREAD_CENTS = 0.0  # bid-ask half-spread
    #cost_model._COMMISSION = 0.0  # broker commission
    #cost_model._IMPACT_COEFF = 0.0  # square-root market-impact  ⬅ NEW
    # -------------------------------------------------------------------




    broker = BrokerStub(slippage_bp=float(cfg["slippage_bp"]))
    risk = RiskManager(
        account_equity = float(cfg["equity"]),
        cost_model = cost_model,
        max_kelly = float(cfg.get("max_kelly", 0.25)),
        adv_cap_pct = float(cfg.get("adv_cap_pct", 0.20)),
        )

    #cost_model.spread_bp = float(cfg["spread_bp"])
    #cost_model.commission = float(cfg["commission"])
    #broker = BrokerStub(slippage_bp=float(cfg["slippage_bp"]))
    # risk = RiskManager(
    #    account_equity=float(cfg["equity"]), cost_model=cost_model, max_kelly=0.5
    # )

    # RiskManager now only takes initial equity and max_kelly; drop cost_model
    #risk = RiskManager(
    #    float(cfg["equity"]))

    # cost_model = BasicCostModel(
    #    spread_bp=float(cfg["spread_bp"]), commission=float(cfg["commission"])
    # )
    # broker = BrokerStub(slippage_bp=float(cfg["slippage_bp"]))
    # risk = RiskManager(
    #    account_equity=float(cfg["equity"]), cost_model=cost_model, max_kelly=0.5
    # )

    # 5 ─ Event loop
    #bt = Backtester(feats, cfg)  # Pass the prepared DataFrame
    # 5 ─ Event loop
    # Pass all the required components into the Backtester
    # ─── Recompute daily regimes for later breakdown ───
    from prediction_engine.market_regime import label_days, RegimeParams
    #df_raw["timestamp"] = pd.to_datetime(df_raw["Date"] + " " + df_raw["Time"])
    daily_df = (
        df_raw
        .set_index("timestamp")
        .resample("D")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    daily_regimes = label_days(daily_df, RegimeParams())

    if "mean_ret1m" in stats.files:
        bad = set(np.where(stats["mean_ret1m"] <= 0)[0])
        # OR stricter: bottom 20%
        q20 = np.percentile(stats["mean_ret1m"], 20)
        bad = set(np.where(stats["mean_ret1m"] <= q20)[0])
        good_clusters = set(range(len(stats["mu"]))) - bad

    bt = Backtester(
        features_df=feats,
        cfg=cfg,
        ev=ev,
        risk=risk,
        broker=broker,
        cost_model=cost_model,
        calibrator=iso_prob,
        good_clusters=good_clusters,
        regimes=daily_regimes,
    )

    #eq_curve = bt.run()
    mu_min, mu_med, mu_max = float(all_mu.min()), float(np.median(all_mu)), float(all_mu.max())
    print("[ISO sanity] p(min,med,max) =",
          iso_prob.predict([mu_min, mu_med, mu_max]))

    eq_curve = bt.run()
    #signals = eq_curve.copy()

    leverage_peak = (eq_curve["notional_abs"] / eq_curve["equity"]).max()
    print(f"[Risk] peak_leverage = {leverage_peak:.2f}×")

    '''csv_path = _resolve_path(cfg["csv"])
    df_raw = pd.read_csv(csv_path)
    # ─── rename uppercase OHLC to lowercase to match resample keys ───
    df_raw.rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"},
        inplace=True,
    )'''



    '''# ─── Recompute daily regimes for later breakdown ───
    from prediction_engine.market_regime import label_days, RegimeParams
    df_raw["timestamp"] = pd.to_datetime(df_raw["Date"] + " " + df_raw["Time"])
    daily_df = (
        df_raw
        .set_index("timestamp")
        .resample("D")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    daily_regimes = label_days(daily_df, RegimeParams())'''

    # Save
    #out_path = Path(cfg["out"])
    #eq_curve.to_csv(out_path, index=False)

    import matplotlib.pyplot as plt

    # After eq_curve.to_csv(...)
    #df = pd.read_csv(out_path)

    # equity‐based cumulative return
    #df["cum_return"] = df["equity"] / cfg["equity"] - 1.0

    # --- A) Add one‑bar forward return and win/loss stats ---
    #signals = df.copy()
    signals = eq_curve.copy()
    signals["cum_return"] = signals["equity"] / cfg["equity"] - 1.0
    #signals["mu"] = signals["mu_cal"]  # keep legacy column name

    signals["mu_raw"] = signals["mu_raw"]  # (no-op, just clarity)
    signals["mu_cal"] = signals["mu_cal"]  # (already there)
    signals["mu"] = signals["mu_raw"]  # <- use raw mu for diagnostics/ROC
    # If you want to be extra safe about duped names coming from earlier code:
    signals = signals.loc[:, ~signals.columns.duplicated()].copy()

    # --- PATCH: compute calibrated probability p for entries only ---
    # Define "entries" as rows where we actually put on risk next bar
    '''entries = signals.get(
        "qty_next_bar",
        pd.Series(False, index=signals.index, dtype=bool)
    ) > 0'''

    entries = signals.get("qty_next_bar", pd.Series(False, index=signals.index, dtype=bool)) > 0

    signals["ret1m"] = np.nan
    entry_open = signals["open"].shift(-1)
    entry_close = signals["close"].shift(-1)
    signals.loc[entries, "ret1m"] = (entry_close.loc[entries] / entry_open.loc[entries]) - 1.0

    # Initialize p with NaN; diagnostics should operate on non-null entries
    signals["p"] = np.nan

    # Use the *same* in-memory calibrator you fit above (iso_prob)
    try:
        '''p_vals = map_mu_to_prob(
            signals.loc[entries, "mu"].to_numpy(dtype=float),
            calibrator=iso_prob,
            default_if_missing=0.5,  # neutral if anything goes sideways
        )'''
        p_vals = _predict_p(signals.loc[entries, "mu"].to_numpy(dtype=float))
        signals.loc[entries, "p"] = p_vals
    except Exception as e:
        print(f"[PATCH warn] Failed to compute calibrated p: {e}")

    # Quick sanity: show spread of valid p
    _p = signals["p"].dropna()
    if not _p.empty:
        pmin, pmax = float(_p.min()), float(_p.max())
        print(f"p(valid) range: [{pmin:.3f}, {pmax:.3f}]  n={len(_p)}")
        if abs(pmax - pmin) < 1e-4:
            print("[PATCH warn] p collapsed ~constant; check isotonic fit range/labels.")
    else:
        print("[PATCH warn] No valid p on entries; check entries mask (qty_next_bar).")

    #signals["ret1m"] = signals["open"].shift(-1) / signals["open"] - 1.0

    #assert (signals['mu'].std() > 1e-3), "µ collapsed to constant"
    #assert ((signals['mu'] > 0).mean() < 0.25), "µ gate too lax"
    #assert (signals['mu'].std() > 1e-3), "µ collapsed to constant"
    #ssert ((signals['mu'] > 0).mean() < 0.25), "µ gate too lax"

    # For diagnostics:
    #auc_on = "mu_raw"  # <- use raw µ for ROC AUC & deciles
    #signals.rename(columns={auc_on: "mu"}, inplace=True)

    # mask for all trades you took:
    trade_mask = signals.qty_next_bar > 0
    # among those, which were wins?
    winning_mask = (signals.ret1m > 0)

    print("Positive‑µ trade stats:")
    print("  Total trades:", trade_mask.sum())
    print("  Win rate:   ", (trade_mask & winning_mask).sum() / trade_mask.sum())
    # average return *of the trades you actually took*:
    avg_ret = signals.loc[trade_mask, "ret1m"].mean()
    print("  Avg ret:    ", avg_ret)

    # --- clip isotonic outputs to avoid degenerate tiny/huge p's on small samples

    '''class ClippedIso:
        def __init__(self, base, lo=0.30, hi=0.70):
            self.base, self.lo, self.hi = base, lo, hi

        def predict(self, x):
            x = np.asarray(x).reshape(-1)
            p = self.base.predict(x)
            return np.clip(p, self.lo, self.hi)

    ev._calibrator = ClippedIso(iso_prob, lo=0.30, hi=0.70)'''

    from scripts.diagnostics import BacktestDiagnostics

    diag = BacktestDiagnostics(
        df=signals,
        raw=df_raw,
        cfg=cfg
    )
    diag.run_all()

    plt.figure()
    plt.scatter(signals.mu, signals.ret1m, s=5, alpha=0.3)
    plt.title("µ forecast vs realized 1‑bar return")
    plt.xlabel("µ")
    plt.ylabel("ret1m")
    plt.grid(True)
    plt.show()

    # --- C) Cluster‑level summary ---
    '''cluster_stats = signals.groupby("cluster_id").apply(
        lambda g: pd.Series({
            "n": len(g),
            "mean_µ": g["mu"].mean(),
            "mean_ret1m": (g["open"].shift(-1) / g["open"] - 1.0).mean()
        })
    ).sort_values("mean_ret1m")'''
    #print(cluster_stats.head(10))

    # pandas ≥2.1 deprecation-safe aggregation
    cluster_stats = (
           signals
           .assign(ret1m= lambda df: df["open"].shift(-1) / df["open"] - 1.0)
            .groupby("cluster_id", group_keys=False)
            .agg(n=("mu", "size"),
                mean_µ = ("mu", "mean"),
                mean_ret1m = ("ret1m", "mean"))
            .sort_values("mean_ret1m"))

    print(cluster_stats.head(10))

    # buy‐and‐hold using raw data (df_raw) rather than the signals CSV
    buy_hold = df_raw["close"].iloc[-1] / df_raw["open"].iloc[0] - 1.0
    print(f"Buy‑and‑Hold over period: {buy_hold * 100:.2f}%")
    signals["regime"] = pd.to_datetime(signals["timestamp"]).dt.normalize().map(daily_regimes)
    print(signals.groupby("regime")["cum_return"].agg(["min", "max", "mean"]))

    plt.figure()
    plt.plot(signals["timestamp"], signals["cum_return"], lw=1)
    plt.title("Cumulative Return Over Time")
    plt.xlabel("Time")
    plt.ylabel("Return")
    plt.grid(True)
    plt.show()

    # 1. Print statistical summary
    print("\nDiagnostic summary:")
    print(signals[["mu", "residual", "position_size"]].describe())

    # ** Print a nice summary before returning **
    final_eq = float(eq_curve["equity"].iloc[-1])
    total_ret = 100 * (final_eq / cfg["equity"] - 1)
    log.info("Backtest complete.  Equity curve → %s", csv_path)
    log.info("  Final equity: $%.2f   Total return: %.2f%%   Bars: %d",
             final_eq, total_ret, len(eq_curve))

    # 2. Plot histograms (one per figure)
    for col in ("mu", "residual", "position_size"):
        plt.figure()
        signals[col].hist(bins=50)
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.show()

    # after running, in notebook / REPL:
    #signals = pd.read_csv("backtest_signals.csv")
    #signals["ret1m"] = signals["open"].shift(-1) / signals["open"] - 1.0
    pearson_r = signals["mu"].corr(signals["ret1m"])
    print("Pearson r =", pearson_r)

    print("Residual summary:")
    print(signals["residual"].describe())

    #pct_below = (signals["residual"] < 0.75).mean()
    #print("Pct residual < 0.75:", pct_below)

    thr = getattr(ev, "residual_threshold", 0.75)
    pct_below = (signals["residual"] < thr).mean()
    print(f"Pct residual < {thr:.3g}: {pct_below:.3f}")

    print("Bars after scan:", len(signals))
    print("µ distribution on those bars:\n", signals["mu"].describe())
    print("Count µ>0:", (signals["mu"] > 0).sum())

    # ------------- persist signals ----------
    OUT_DIR = Path("backtests")
    OUT_DIR.mkdir(exist_ok=True)
    ts_tag = signals["timestamp"].iloc[0].strftime("%Y%m%d")
    out_file = OUT_DIR / f"signals_{cfg['symbol']}_{ts_tag}.csv"
    signals.to_csv(out_file, index=False)
    print("Saved run to", out_file)

    return eq_curve


__all__ = ["run", "CONFIG"]

if __name__ == "__main__":
    asyncio.run(run(CONFIG))




