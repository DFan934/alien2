# ---------------------------------------------------------------------------
# FILE: scripts/tune_hyper.py
# ---------------------------------------------------------------------------
"""Bayesian search for EVEngine hyper‑parameters (k, h, α).

Uses *yesterday’s* live features + realised returns that are written by the
feed‑handler into ``data/daily_signals.csv``.  Produces
``artifacts/best_hyper.yml`` consumed at engine reload time.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
from skopt.space import Real, Integer


# Dummy wrapper – replace with real back‑tester later
class _EVBacktest(BaseEstimator):
    def __init__(self, k: int = 16, h: float = 1.0, alpha: float = 0.5):
        self.k = k
        self.h = h
        self.alpha = alpha

    def fit(self, X, y):  # noqa: N802 – sklearn contract
        return self  # no‑op – stateless wrapper

    def score(self, X, y):  # noqa: N802
        # toy Sharpe: mean / std
        ret = y  # y already realised returns per sample
        mu, sig = float(ret.mean()), float(ret.std(ddof=0)) + 1e-8
        return mu / sig


_DATA_CSV = Path("data/daily_signals.csv")
_OUT_YAML = Path("artifacts/best_hyper.yml")


def optimise():
    df = pd.read_csv(_DATA_CSV, parse_dates=[0])
    X = df.drop(columns=["realised_ret"]).values  # not really used
    y = df["realised_ret"].values

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
    )
    opt.fit(X, y)

    best = opt.best_params_
    _OUT_YAML.parent.mkdir(parents=True, exist_ok=True)
    _OUT_YAML.write_text("\n".join(f"{k}: {v}" for k, v in best.items()))
    print("Best hyper‑params →", best)


if __name__ == "__main__":
    optimise()
