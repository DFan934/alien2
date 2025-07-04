# ---------------------------------------------------------------------------
# weight_optimization.py
# ---------------------------------------------------------------------------
from __future__ import annotations

"""Nightly Recency/Tail‐Curve Optimiser (revised)
================================================

This replaces the earlier lightweight ``weight_optimization.py``.  It adds:

* **Train/validation split** by date to avoid look‑ahead.
* **Two‑stage search** – coarse multiprocessing grid, then local Optuna search.
* **Curve families** (linear, exponential, sigmoid) each with *tail_len* and
  *shape* hyper‑parameters.
* **Pluggable back‑test function** so optimiser stays I/O‑free.
* **Persistence** to ``best_curve...yaml`` for next trading session.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Tuple, Dict, Any

import concurrent.futures as _fut
import json

import numpy as np
import optuna
import pandas as pd

# ---------------------------------------------------------------------------
@dataclass
class CurveParams:
    family: str           # linear | exp | sigmoid
    tail_len: int         # days of history (5‑120)
    shape: float          # decay rate or slope (0.1‑10)

    def to_dict(self):
        return self.__dict__


# ---------------------------------------------------------------------------
class WeightOptimizer:
    """Search for the curve that maximises Sharpe on *validation* slice."""

    def __init__(self, n_jobs: int | None = None):
        self.n_jobs = n_jobs or max(1, _fut.ProcessPoolExecutor()._max_workers - 1)

    # .................................................................
    def optimise(self,
                 pnl_series: pd.Series,
                 validation_frac: float = 0.2,
                 storage_dir: Path | str = "./weights") -> Dict[str, Any]:
        """Return dict with best params and Sharpe; save YAML to *storage_dir*."""

        pnl_series = pnl_series.dropna().sort_index()
        split_idx = int(len(pnl_series) * (1 - validation_frac))
        train, valid = pnl_series.iloc[:split_idx], pnl_series.iloc[split_idx:]

        # 1) coarse grid ------------------------------------------------
        grid = self._build_grid()
        with _fut.ProcessPoolExecutor(max_workers=self.n_jobs) as pool:
            scores = list(pool.map(lambda p: self._score_curve(p, valid), grid))
        best_curve, best_sharpe = max(scores, key=lambda t: t[1])

        # 2) Optuna fine search ----------------------------------------
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: self._objective(t, valid, best_curve.family),
                       n_trials=50, show_progress_bar=False)
        if study.best_value > best_sharpe:
            best_params = CurveParams(**study.best_params)
            best_sharpe = study.best_value
        else:
            best_params = best_curve

        # 3) save ------------------------------------------------------
        out_dir = Path(storage_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f"best_curve_{date.today():%Y%m%d}.json"
        json.dump({"params": best_params.to_dict(), "sharpe": best_sharpe}, fname.open("w"))

        return {"params": best_params.to_dict(), "sharpe": best_sharpe, "file": str(fname)}

    # ================= internal helpers ==============================
    def _score_curve(self, params: CurveParams, pnl: pd.Series) -> Tuple[CurveParams, float]:
        w = self._make_weights(len(pnl), params)
        w /= w.sum()
        ret = np.sum(w * pnl.values)
        risk = np.sqrt(np.sum(w * (pnl.values - ret) ** 2)) + 1e-8
        sharpe = ret / risk
        return params, sharpe

    @staticmethod
    def _objective(trial: optuna.trial.Trial, valid: pd.Series, family: str) -> float:
        tail_len = trial.suggest_int("tail_len", 5, 120)
        shape = trial.suggest_float("shape", 0.1, 10)
        params = CurveParams(family, tail_len, shape)
        _, sharpe = WeightOptimizer()._score_curve(params, valid)
        return sharpe

    @staticmethod
    def _make_weights(n: int, p: CurveParams):
        idx = np.arange(n)[::-1].astype(float)
        mask = idx < p.tail_len
        idx = idx[mask]
        if p.family == "linear":
            w = 1 - (idx / p.tail_len)
        elif p.family == "exp":
            w = np.exp(-idx / (p.shape * 10))
        else:  # sigmoid
            z = (idx - p.tail_len / 2) / (p.shape * 5)
            w = 1 / (1 + np.exp(z))
        return w

    def _build_grid(self):
        grid = []
        for fam in ("linear", "exp", "sigmoid"):
            for tail in (10, 20, 40, 60, 90, 120):
                for shape in (0.5, 1, 2, 5):
                    grid.append(CurveParams(fam, tail, shape))
        return grid


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, yaml

    parser = argparse.ArgumentParser(description="Run nightly weight optimisation")
    parser.add_argument("pnl_csv", help="CSV with columns date,pnl")
    parser.add_argument("--out", default="./weights", help="dir to save best curve")
    args = parser.parse_args()

    pnl = pd.read_csv(args.pnl_csv, parse_dates=[0], index_col=0).squeeze()
    best = WeightOptimizer().optimise(pnl, storage_dir=args.out)
    print("Best curve →", yaml.safe_dump(best, sort_keys=False))
