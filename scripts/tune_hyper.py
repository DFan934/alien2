# ---------------------------------------------------------------------------
# FILE: scripts2/tune_hyper.py
# ---------------------------------------------------------------------------
"""
Bayesian search for EVEngine hyper-parameters (k, h, α).

It back-tests on *yesterday’s* live features and realised returns found in
``data/daily_signals.csv``.  Each sample must contain:

* ``realised_ret``   – realised $/share P&L
* ``adv_pct``        – %ADV traded (0-100).  Used by the cost model.

Outputs the best triple to ``artifacts/best_hyper.yml``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
from skopt.space import Real, Integer

from prediction_engine.tx_cost import BasicCostModel

_COST = BasicCostModel()

# ---------------------------------------------------------------------------
# Back-test wrapper (stateless, fast)
# ---------------------------------------------------------------------------
class _EVBacktest(BaseEstimator):
    """
    Lightweight wrapper that evaluates cost-adjusted Sharpe-ratio
    for a given (k, h, alpha) triple.  It does *not* simulate fills
    or Kelly sizing – the intent is to rank hyper-parameters quickly.
    """

    def __init__(self, k: int = 16, h: float = 1.0, alpha: float = 0.5):
        self.k = k
        self.h = h
        self.alpha = alpha

    # sklearn API – no training needed --------------------------------
    def fit(self, X, y):  # noqa: N802
        return self

    # -----------------------------------------------------------------
    def score(self, X, y):  # noqa: N802
        """
        X[:, 0] : adv_pct  (float 0-100)
        y       : realised return per share (already signed)
        """
        adv_pct = X[:, 0]
        costs = np.fromiter(
            (_COST.estimate(adv_percentile=p) for p in adv_pct),
            dtype=np.float32,
            count=len(adv_pct),
        )
        net_ret = y - costs
        mu = float(net_ret.mean())
        sig = float(net_ret.std(ddof=0)) + 1e-8
        return mu / sig  # Sharpe proxy


_DATA_CSV = Path("data/daily_signals.csv")
_OUT_YAML = Path("artifacts/best_hyper.yml")


def optimise() -> None:
    # -----------------------------------------------------------------
    # 1. Load CSV  (must contain realised_ret  &  adv_pct)
    # -----------------------------------------------------------------
    df = pd.read_csv(_DATA_CSV, parse_dates=[0])
    if not {"realised_ret", "adv_pct"}.issubset(df.columns):
        raise RuntimeError("daily_signals.csv missing required columns")

    X = df[["adv_pct"]].to_numpy(dtype=np.float32)
    y = df["realised_ret"].to_numpy(dtype=np.float32)

    # -----------------------------------------------------------------
    # 2. Bayesian optimisation over (k, h, alpha)
    # -----------------------------------------------------------------
    param_space = {
        "k": Integer(4, 64),
        "h": Real(0.3, 4.0, prior="log-uniform"),
        "alpha": Real(0.0, 1.0),
    }
    cv = TimeSeriesSplit(n_splits=4)
    opt = BayesSearchCV(
        _EVBacktest(),
        param_space,
        n_iter=25,
        scoring=make_scorer(lambda est, X, y: est.score(X, y)),
        cv=cv,
        n_jobs=-1,
        verbose=0,
    )
    opt.fit(X, y)

    best = opt.best_params_
    _OUT_YAML.parent.mkdir(parents=True, exist_ok=True)
    _OUT_YAML.write_text("\n".join(f"{k}: {v}" for k, v in best.items()))
    print("Best hyper-params →", best)


if __name__ == "__main__":
    optimise()
