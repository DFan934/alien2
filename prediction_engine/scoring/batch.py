# prediction_engine/scoring/batch.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass(frozen=True)
class BatchScoreResult:
    # columns: ['timestamp','symbol','p_raw','p_cal'] (at minimum)
    frame: pd.DataFrame

def score_batch(ev_engine, X: np.ndarray) -> np.ndarray:
    """
    Single vectorized call into EVEngine (or compatible) that returns probabilities
    for a batch of feature rows. Assumes ev_engine exposes predict_proba(X).
    """
    # expected shape: (N, F)
    return np.asarray(ev_engine.predict_proba(X)).reshape(-1)

def score_per_symbol_loop(ev_engine, rows: list[np.ndarray]) -> np.ndarray:
    """
    Baseline: one predict_proba call *per row* (simulates old per-symbol/per-minute loop).
    """
    out = []
    for x in rows:
        out.append(float(ev_engine.predict_proba(x.reshape(1, -1)).ravel()[0]))
    return np.asarray(out, dtype=float)

def vectorize_minute_batch(ev_engine, df_batch: pd.DataFrame, pca_cols: list[str]) -> BatchScoreResult:
    """
    df_batch: rows for ONE minute across MANY symbols (each row one symbol).
      Must include: ['timestamp','symbol', *pca_cols]
    Returns a DataFrame with the same order + ['p_raw','p_cal'] (p_cal= p_raw passthrough here;
    your calibrator may post-process later in the pipeline).
    """
    X = df_batch[pca_cols].to_numpy(dtype=float, copy=False)
    p = score_batch(ev_engine, X)
    out = df_batch[["timestamp", "symbol"]].copy()
    out["p_raw"] = p
    # leave calibration to the existing calibrator step; passthrough here
    out["p_cal"] = out["p_raw"]
    return BatchScoreResult(out)
