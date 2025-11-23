# prediction_engine/scoring/batch.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from prediction_engine.calibration import map_mu_to_prob
from prediction_engine.market_regime import MarketRegime

@dataclass(frozen=True)
class BatchScoreResult:
    # columns: ['timestamp','symbol','p_raw','p_cal'] (at minimum)
    frame: pd.DataFrame

def score_batch(ev_engine, X, regimes=None):
    """
    Returns calibrated probabilities for each row in X.
    Works with engines that expose predict_proba(X) OR EVEngine.evaluate(x, regime=...).
    """
    # If the engine already has the scikit-like API, use it.
    if hasattr(ev_engine, "predict_proba"):
        return np.asarray(ev_engine.predict_proba(X)).reshape(-1)

    # Otherwise, assume EVEngine-style API: evaluate(x, regime) → object with .mu
    n = len(X)
    if regimes is None:
        reg_list = [MarketRegime.TREND] * n
    else:
        # regimes can be strings or enums; normalize to MarketRegime
        reg_list = []
        for r in regimes:
            if isinstance(r, MarketRegime):
                reg_list.append(r)
            else:
                try:
                    reg_list.append(MarketRegime.from_string(str(r)))
                except Exception:
                    try:
                        reg_list.append(MarketRegime[str(r).upper()])
                    except Exception:
                        reg_list.append(MarketRegime.TREND)

    mus = np.empty(n, dtype=float)
    for i, (x, reg) in enumerate(zip(X, reg_list)):
        mus[i] = ev_engine.evaluate(x, regime=reg).mu

    # If EVEngine carries a calibrator internally, great; otherwise pass None.
    calibrator = getattr(ev_engine, "calibrator", None)
    return map_mu_to_prob(mus, calibrator=calibrator)

def score_per_symbol_loop(ev_engine, rows: list[np.ndarray]) -> np.ndarray:
    """
    Baseline loop scorer that works with either a scikit-like model exposing
    predict_proba(X) or an EVEngine exposing evaluate(x, regime=...).
    Returns calibrated probabilities when EVEngine is used.
    """
    # Fast path: scikit-style
    if hasattr(ev_engine, "predict_proba"):
        return np.asarray(ev_engine.predict_proba(np.vstack([r.reshape(1, -1) for r in rows]))).reshape(-1)

    # EVEngine path: evaluate() -> .mu, then map μ→p using engine.calibrator if present
    out_mu = np.empty(len(rows), dtype=float)
    for i, x in enumerate(rows):
        out_mu[i] = ev_engine.evaluate(x, regime=MarketRegime.TREND).mu  # default TREND; no per-row regimes here
    calibrator = getattr(ev_engine, "calibrator", None)
    return map_mu_to_prob(out_mu, calibrator=calibrator)


def vectorize_minute_batch(ev_engine, df_batch: pd.DataFrame, pca_cols: list[str]) -> BatchScoreResult:
    """
    df_batch: rows for ONE minute across MANY symbols (each row one symbol).
      Must include: ['timestamp','symbol', *pca_cols]
    Produces calibrated probabilities (p_raw==p_cal here). If you have a later
    calibration step, keep it a no-op to avoid double-calibration.
    """
    X = df_batch[pca_cols].to_numpy(dtype=float, copy=False)

    # Thread regimes if available; else default TREND
    regimes = df_batch["regime"].astype(str).tolist() if "regime" in df_batch.columns else None

    p = score_batch(ev_engine, X, regimes=regimes)  # already calibrated if EVEngine

    out = df_batch[["timestamp", "symbol"]].copy()
    out["p_raw"] = p
    out["p_cal"] = out["p_raw"]  # keep passthrough to avoid double-cal
    return BatchScoreResult(out)
