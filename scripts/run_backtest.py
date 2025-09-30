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
from scripts.a2_report import generate_report  # NEW
from sklearn.isotonic import IsotonicRegression

import logging
import sys
from pathlib import Path
from typing import Any, Dict
from prediction_engine.testing_validation.walkforward import WalkForwardRunner

from prediction_engine.calibration import load_calibrator, map_mu_to_prob

#import pandas as pd

from feature_engineering.pipelines.core import CoreFeaturePipeline
from prediction_engine.ev_engine import EVEngine
from prediction_engine.tx_cost import BasicCostModel  # NEW
from execution.risk_manager import RiskManager
from prediction_engine.testing_validation.async_backtester import BrokerStub  # NEW
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

    "horizon_bars": 20,
    "longest_lookback_bars": 60,
    "p_gate_quantile": 0.55,
    "full_p_quantile": 0.65,
    "artifacts_root": "artifacts/a2",   # where per-fold outputs go

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
    # debug/test toggle
    "debug_no_costs": True,  # ← set True for the tiny RRC slice
# dev gating options
    "dev_scanner_loose": True,    # ← NEW
    "dev_detector_mode": "OR",    # ← NEW; force OR even without YAML
    "sign_check": True,           # ← NEW; report will compute AUC(1−p)
    "min_entries_per_fold": 100,  # ← NEW; fail fast if < 100 entries
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
        level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout
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
    # Keep a copy of the full (unscanned) bars for A2 folds/labels


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

    #detector = build_detectors()
    # BEFORE:
    # detector = build_detectors()
    # AFTER:
    detector = build_detectors(dev_loose=bool(cfg.get("dev_scanner_loose", False)))
    # Force OR mode in dev if requested
    if cfg.get("dev_detector_mode", "").upper() in {"OR", "AND"}:
        detector.mode = cfg["dev_detector_mode"].upper()

    # Scan (async)
    mask = await detector(df_raw)
    pass_rate = mask.mean() * 100
    print(f"[Scanner KPI] Bars passing filters: {mask.sum()} / {len(mask)} = {pass_rate:.2f}%")

    df_full = df_raw.copy()

    # This is the asynchronous call that needs 'await'
    #mask = await detector(df_raw)

    #pass_rate = mask.mean() * 100
    #print(f"[Scanner KPI] Bars passing filters: {mask.sum()} / {len(mask)} = {pass_rate:.2f}%")

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

    # === A2: Purged walk-forward, train-only calibration & gates ===
    from pathlib import Path as _P

    resolved_parquet_root = _resolve_path(cfg["parquet_root"])

    runner = WalkForwardRunner(
        artifacts_root=_P(cfg.get("artifacts_root", "artifacts/a2")),
        #parquet_root=_P(cfg.get("parquet_root", "parquet")),
        parquet_root=resolved_parquet_root,
        ev_artifacts_root=_P(cfg["artefacts"]),
        symbol=cfg["symbol"],
        horizon_bars=int(cfg.get("horizon_bars", 20)),
        longest_lookback_bars=int(cfg.get("longest_lookback_bars", 60)),
        p_gate_q=float(cfg.get("p_gate_quantile", 0.65)),
        full_p_q=float(cfg.get("full_p_quantile", 0.80)),
        debug_no_costs=bool(cfg.get("debug_no_costs", False)),

    )

    # --- Persist label metadata for A2 report & downstream consumers ---
    meta_path = _resolve_path(cfg.get("artifacts_root", "artifacts/a2")) / "meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    label_meta = {
        "label_horizon": f"open→open log-return, H={int(cfg['horizon_bars'])} bars",
        "label_function": "feature_engineering.labels.labeler.one_bar_ahead",
        "label_side": "long_positive",  # y=1 iff ret_fwd > 0
        "label_threshold": 0.0,  # decision boundary for classification
        "sign_check": bool(cfg.get("sign_check", False)),
        "min_entries_per_fold": int(cfg.get("min_entries_per_fold", 0)),
        "scanner_dev_loose": bool(cfg.get("dev_scanner_loose", False)),
        "detector_mode": detector.mode,
    }
    # merge into existing RUN_META for traceability

    label_meta["use_isotonic"] = bool(cfg.get("use_isotonic", True))

    meta_doc = {**RUN_META, **label_meta}
    meta_path.write_text(json.dumps(meta_doc, indent=2))

    # ── µ→p isotonic calibration helper for WalkForwardRunner ──
    def _fit_isotonic_from_val(mu_val: np.ndarray, y_val: np.ndarray, *, use_iso: bool):
        """
        Fit IsotonicRegression mapping local mean return mu_k → P(y=1).
        Returns a sklearn calibrator or None.
        """
        if not use_iso:
            return None
        mu_val = np.asarray(mu_val).reshape(-1, 1)
        y_val = np.asarray(y_val).astype(float).ravel()
        # guards: enough samples + class diversity
        if mu_val.shape[0] < 50 or len(np.unique(y_val[~np.isnan(y_val)])) < 2:
            return None
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        iso.fit(mu_val, y_val)
        return iso

    '''_ = runner.run(
        df_full=df_full,
        df_scanned=df_raw,  # already scanner-filtered bars above
        start=cfg["start"],
        end=cfg["end"],
    )'''

    _ = runner.run(
        df_full=df_full,
        df_scanned=df_raw,
        start=cfg["start"],
        end=cfg["end"],
        # NEW: let the runner ask us to fit an iso calibrator from (mu_val, y_val)
        calibrator_fn=lambda mu_val, y_val: _fit_isotonic_from_val(
            mu_val, y_val, use_iso=bool(cfg.get("use_isotonic", True))
        ),
        # NEW: pass through EVEngine build kwargs the runner should use for TEST
        ev_engine_overrides={
            "metric": cfg.get("metric", "mahalanobis"),
            "k": int(cfg.get("k_max", 64)),
            "residual_threshold": float(cfg.get("residual_threshold", 0.75)),
        },
        # (optional) ask runner to persist both raw & calibrated p for the report
        persist_prob_columns=("p_raw", "p_cal"),
    )

    # === Post-A2 report (no leakage, folds aggregated) ===
    # where you already call generate_report(...)
    generate_report(
        artifacts_root=cfg.get("artifacts_root", "artifacts/a2"),
        csv_path=_resolve_path(cfg["csv"]),  # ← use the resolver you already have in run_backtest.py
    )

    # Optional: return a small summary for callers; __main__ ignores this anyway.
    return pd.DataFrame([{"run_id": RUN_ID, "out_dir": str(cfg.get("artifacts_root", "artifacts/a2"))}])


__all__ = ["run", "CONFIG"]

if __name__ == "__main__":
    asyncio.run(run(CONFIG))