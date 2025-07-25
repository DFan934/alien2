# ---------------------------------------------------------------------------
# market_regime.py
# ---------------------------------------------------------------------------
from __future__ import annotations

"""Market Regime Detector (upgraded)
===================================

Implements the blueprint‑required classification of each **trading day** into
one of three canonical regimes:

* ``TRENDING``   – Strong directional movement, low relative volatility
* ``CHOPPY``     – Sideways / range‑bound, moderate volatility
* ``VOLATILE``   – Large intraday swings, volatility spike

Two technical measures are added to the original VWAP‑slope heuristic:

* **ADX14**  (Average Directional Index, from DMI)
* **ATR‑z**  (14‑period ATR z‑score vs 100‑day rolling mean)

The detector consumes a *daily* OHLCV DataFrame indexed by date and returns a
``pd.Series[str]`` of regime labels.  A streaming version (`stream_labels`)
updates labels bar‑by‑bar for intraday use.

------------------------------------------------------------------------
Blueprint mapping
------------------------------------------------------------------------
* DMI/ADX satisfies “metrics like DMI” req.
* ATR‑z covers “ATR” & volatility spike logic.
* Regime shift fallback: when label changes the helper keeps a rolling memory
  of the previous label to smooth whipsaws.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple, Optional

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
#   Light, intraday regime detector (Task 4)  ✨
# ----------------------------------------------------------------------
from enum import Enum, auto
import pickle, time
from pathlib import Path

class MarketRegime(Enum):
    TREND   = auto()
    RANGE   = auto()
    VOLATILE = auto()

_STATE_PATH = Path("state/market_regime.pkl")

class RegimeDetector:
    """
    Streaming detector that recalculates every call to **update()** on the
    most-recent bars (expects columns high/low/close).  The mapping rules are
        ADX > 25   &  RV < 0.75×median  → TREND
        ADX < 20   &  RV < 1.25×median → RANGE
        otherwise                         VOLATILE
    The latest label is pickled every 15 minutes to *_STATE_PATH* so any
    worker process (e.g. nightly calibration) can read it.
    """

    def __init__(self) -> None:
        self.state: MarketRegime = MarketRegime.RANGE
        self._rv_median: float   = 0.0
        self._last_save: float   = 0.0

    # ---- public -----------------------------------------------------
    def update(self, bar_df: pd.DataFrame) -> MarketRegime:
        # ------------------------------------------------------------------
        # graceful fallback for synthetic unit-test bars                    ← NEW
        # ------------------------------------------------------------------
        if {"price"}.issubset(bar_df.columns) and not {"close"}.issubset(bar_df.columns):
            bar_df = bar_df.assign(close=bar_df["price"],
                                   high=bar_df["price"],
                                   low=bar_df["price"])

        df = bar_df.tail(60).copy()
        adx = _dmi(df, 14)[2].iloc[-1]


        # --- inside RegimeDetector.update() ---------------------------------
        # ------------------------------------------------------------------
        # inside RegimeDetector.update()
        # ------------------------------------------------------------------
        rv = df["close"].pct_change().rolling(5).std() * np.sqrt(5)
        rv_latest = float(rv.iloc[-1])
        rv_median = rv.median() if self._rv_median == 0 else self._rv_median
        self._rv_median = 0.95 * rv_median + 0.05 * rv_latest

        ratio = rv_latest / (self._rv_median + 1e-9)

        # ---- final mapping ------------------------------------------------
        if ratio >= 2.5:  # big noise spike ⇒ VOLATILE  (override)
            self.state = MarketRegime.VOLATILE
        elif adx >= 20:  # clear direction ⇒ TREND
            self.state = MarketRegime.TREND
        else:  # everything else ⇒ RANGE
            self.state = MarketRegime.RANGE

        # persist every 15 min
        now = time.time()
        if now - self._last_save > 900:
            _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_STATE_PATH, "wb") as fh:
                pickle.dump({"ts": now, "state": self.state}, fh)
            self._last_save = now

        # ──--- DEBUG – remove after we’re happy ───────────────
        print(f"[DEBUG] rv_latest={rv_latest:.5f} "
              f"rv_median={self._rv_median:.5f} "
              f"ratio={ratio:.3f}  → {self.state.name}")
        # ───────────────────────────────────────────────────────
        return self.state





# ---------------------------------------------------------------------
class Regime(str, Enum):
    TRENDING = "TRENDING"
    CHOPPY = "CHOPPY"
    VOLATILE = "VOLATILE"

# ---------------------------------------------------------------------
@dataclass
class RegimeParams:
    adx_window: int = 14
    atr_window: int = 14
    atr_z_lookback: int = 100  # for z‑score baseline
    adx_trend_th: float = 25.0
    atr_vol_th: float = 1.0    # z‑score threshold
    debounce_days: int = 3     # avoid regime flapping


# ---------------------------------------------------------------------
# ----  Indicator helpers  --------------------------------------------

def _true_range(df: pd.DataFrame) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    return pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)


def _dmi(df: pd.DataFrame, window: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    plus_dm = df["high"].diff()
    minus_dm = df["low"].diff().abs()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = _true_range(df)
    atr = tr.rolling(window).mean()
    plus_di = 100 * (plus_dm.rolling(window).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window).sum() / atr)
    adx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di)).rolling(window).mean()
    return plus_di, minus_di, adx


def _atr(df: pd.DataFrame, window: int) -> pd.Series:
    tr = _true_range(df)
    return tr.rolling(window).mean()

# ---------------------------------------------------------------------
# ----  Public API  ----------------------------------------------------

def label_days(df: pd.DataFrame, params: RegimeParams | None = None) -> pd.Series:
    """Return a Series of regime labels indexed like *df*.

    *df* must provide columns ``high``, ``low``, ``close``.
    """
    if params is None:
        params = RegimeParams()

    _, _, adx = _dmi(df, params.adx_window)
    atr = _atr(df, params.atr_window)
    atr_mean = atr.rolling(params.atr_z_lookback).mean()
    atr_std = atr.rolling(params.atr_z_lookback).std(ddof=0)
    atr_z = (atr - atr_mean) / (atr_std + 1e-9)

    labels = pd.Series(Regime.CHOPPY, index=df.index, dtype="object")

    labels = labels.where(~((adx >= params.adx_trend_th) & (atr_z < params.atr_vol_th)), Regime.TRENDING)
    labels = labels.where(~(atr_z >= params.atr_vol_th), Regime.VOLATILE)

    # Debounce to avoid rapid flip‑flops
    for i in range(1, len(labels)):
        if labels.iloc[i] != labels.iloc[i - 1]:
            # look back last debounce_days; if predominant label diff, keep, else revert
            start = max(0, i - params.debounce_days)
            window = labels.iloc[start:i]
            if (window == labels.iloc[i]).sum() < params.debounce_days // 2:
                labels.iloc[i] = labels.iloc[i - 1]

    return labels


# Streaming variant ----------------------------------------------------

def stream_labels(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    params: RegimeParams | None = None,
    prev_state: Optional[dict] = None,
) -> Tuple[Regime, dict]:
    """Incremental regime labelling for intraday loops.

    Parameters
    ----------
    highs, lows, closes
        Latest *N* bar arrays (newest last).  N must be >= ``params.atr_window``.
    prev_state
        Dict returned by last call; maintains moving sums for speed.

    Returns
    -------
    label : Regime
    state : dict (pass into next call)
    """
    import collections

    if params is None:
        params = RegimeParams()

    N = len(closes)
    w = params.atr_window
    assert N >= w, "Not enough bars for ATR/DMI"

    if prev_state is None:
        # cold start: compute indicators full‑window
        df = pd.DataFrame({"high": highs, "low": lows, "close": closes})
        label = label_days(df.tail(2 * params.atr_z_lookback), params).iloc[-1]
        state = {
            "adx": _dmi(df, w)[2].iloc[-1],
            "atr": _atr(df, w).iloc[-1],
            "atr_mean": _atr(df, w).tail(params.atr_z_lookback).mean(),
            "atr_std": _atr(df, w).tail(params.atr_z_lookback).std(ddof=0),
            "last_label": label,
            "deque": collections.deque(labels := label_days(df.tail(params.debounce_days), params), maxlen=params.debounce_days),
        }
        return label, state

    # incremental update ------------------------------------------------
    # simplistic; recompute last window only
    hi, lo, cl = highs[-w:], lows[-w:], closes[-w:]
    tr = np.maximum.reduce([
        hi - lo,
        np.abs(hi - closes[-w - 1:-1]),
        np.abs(lo - closes[-w - 1:-1]),
    ])
    atr = tr.mean()

    # ADX approximation: reuse prev plus_dm/minus_dm sums not stored here → fallback to full compute every bar is acceptable for minute‑bar batch sizes
    df_tail = pd.DataFrame({"high": highs[-2 * w:], "low": lows[-2 * w:], "close": closes[-2 * w:]})
    adx_new = _dmi(df_tail, w)[2].iloc[-1]

    # update z‑score baseline
    atr_mean = (prev_state["atr_mean"] * (params.atr_z_lookback - 1) + atr) / params.atr_z_lookback
    delta = atr - prev_state["atr_mean"]
    atr_std = np.sqrt(((params.atr_z_lookback - 1) * prev_state["atr_std"] ** 2 + delta ** 2) / params.atr_z_lookback)
    atr_z = (atr - atr_mean) / (atr_std + 1e-9)

    # decide label ------------------------------------------------------
    label = Regime.CHOPPY
    if adx_new >= params.adx_trend_th and atr_z < params.atr_vol_th:
        label = Regime.TRENDING
    elif atr_z >= params.atr_vol_th:
        label = Regime.VOLATILE

    # debounce
    dq = prev_state["deque"]
    dq.append(label)
    if dq.count(label) < params.debounce_days // 2:
        label = prev_state["last_label"]
    prev_state.update({
        "adx": adx_new,
        "atr": atr,
        "atr_mean": atr_mean,
        "atr_std": atr_std,
        "last_label": label,
    })
    return label, prev_state


# ---------------------------------------------------------------------
if __name__ == "__main__":  # quick smoke test
    import yfinance as yf

    spy = yf.download("SPY", period="1y", interval="1d").rename(columns=str.lower)
    labels = label_days(spy)
    print(labels.tail())
