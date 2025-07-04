# ---------------------------------------------------------------------------
# hyperparam_tuning.py
# ---------------------------------------------------------------------------
"""Async hyper‑parameter tuning (grid + Bayes)."""
import asyncio
from functools import partial
from typing import Dict

import numpy as np
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV  # type: ignore


async def run_grid_async(estimator, param_grid: Dict, X, y, cv=3):
    loop = asyncio.get_running_loop()
    gscv = GridSearchCV(estimator, param_grid, cv=cv, n_jobs=-1)
    return await loop.run_in_executor(None, gscv.fit, X, y)


def run_bayes_opt(estimator, search_space: Dict, X, y, cv=3):  # noqa: D401
    bscv = BayesSearchCV(estimator, search_space, cv=cv, n_iter=50, n_jobs=-1)
    return bscv.fit(X, y)
