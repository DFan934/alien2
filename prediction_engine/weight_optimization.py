# ---------------------------------------------------------------------------
# FILE: prediction_engine/weight_optimization.py
# ---------------------------------------------------------------------------
"""Recency‑curve optimiser with train/valid/test split + Optuna fine search.

The optimiser produces **one JSON per market‑regime** so EVEngine can apply
ageing weights dynamically.  A curve is accepted only if
``test_sharpe >= 0.8 * train_sharpe`` to curb over‑fitting.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Tuple, Literal

import numpy as np
import optuna
import pandas as pd
from typing import IO


RegimeT = Literal["trend", "range", "volatile", "global"]


@dataclass(slots=True)
class CurveParams:
    family: Literal["linear", "exp", "sigmoid"]
    tail_len: int      # days
    shape: float       # decay / slope

    def to_dict(self):
        return self.__dict__


# ---------------------------------------------------------------------------
class WeightOptimizer:
    """Optimises ageing‑weight curve for a single *regime* PnL series."""

    def __init__(self, n_jobs: int | None = None):
        #self.n_jobs = n_jobs or max(1, (cf.ProcessPoolExecutor()._max_workers - 1))
        cpu = max(2, (os.cpu_count() or 2))
        self.n_jobs = n_jobs or max(1, cpu - 1)

    # .....................................................................
    def optimise(
        self,
        pnl: pd.Series,
        *,
        regime: RegimeT,
        validation_frac: float = 0.2,
        test_frac: float = 0.1,
        out_dir: Path | str = Path("weights/recency_curves"),
    ) -> Dict[str, float]:
        """Return dict with train/valid/test Sharpe and persist JSON."""
        pnl = pnl.dropna().sort_index()
        n = len(pnl)
        n_train = int(n * (1 - validation_frac - test_frac))
        n_valid = int(n * validation_frac)
        train, valid, test = (
            pnl.iloc[:n_train],
            pnl.iloc[n_train : n_train + n_valid],
            pnl.iloc[n_train + n_valid :],
        )

        # ---------- coarse grid --------------------------------------
        grid = self._grid()
        with cf.ProcessPoolExecutor(max_workers=self.n_jobs) as pool:
            scores = list(pool.map(lambda p: self._score(p, valid), grid))
        best_p, best_sh_valid = max(scores, key=lambda t: t[1])

        # ---------- Optuna refine ------------------------------------
        study = optuna.create_study(direction="maximize")

        def _obj(trial: optuna.Trial):
            cand = CurveParams(
                family=best_p.family,
                tail_len=trial.suggest_int("tail_len", 5, 120),
                shape=trial.suggest_float("shape", 0.1, 10),
            )
            return self._score(cand, valid)[1]

        study.optimize(_obj, n_trials=64, show_progress_bar=False)
        p_opt = CurveParams(**study.best_params)
        sharpe_valid = study.best_value
        if sharpe_valid < best_sh_valid:
            p_opt = best_p
            sharpe_valid = best_sh_valid

        sharpe_train = self._score(p_opt, train)[1]
        sharpe_test = self._score(p_opt, test)[1]

        if sharpe_test < 0.8 * sharpe_train:
            raise RuntimeError("Curve over‑fits – test Sharpe too low vs train")

        # ---------- persist -----------------------------------------
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fname = Path(out_dir) / f"best_curve_{regime}_{date.today():%Y%m%d}.json"
        with fname.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "params": p_opt.to_dict(),
                    "train_sharpe": sharpe_train,
                    "valid_sharpe": sharpe_valid,
                    "test_sharpe": sharpe_test,
                },
                fh,
                indent=2,
            )

        return {
            "file": str(fname),
            "train": sharpe_train,
            "valid": sharpe_valid,
            "test": sharpe_test,
        }

    # ================= helpers ======================================
    def _score(self, p: CurveParams, pnl: pd.Series) -> Tuple[CurveParams, float]:
        w = self._weights(len(pnl), p)
        w /= w.sum()
        ret = np.dot(w, pnl.values)
        risk = np.sqrt(np.dot(w, (pnl.values - ret) ** 2)) + 1e-9
        return p, ret / risk

    @staticmethod
    def _weights(n: int, p: CurveParams):
        idx = np.arange(n, dtype=float)[::-1]
        mask = idx < p.tail_len
        idx = idx[mask]
        if p.family == "linear":
            w = 1 - idx / p.tail_len
        elif p.family == "exp":
            w = np.exp(-idx / (p.shape * 10))
        else:  # sigmoid
            z = (idx - p.tail_len / 2) / (p.shape * 5)
            w = 1 / (1 + np.exp(z))
        return w

    @staticmethod
    def _grid() -> list[CurveParams]:
        families: tuple[Literal["linear", "exp", "sigmoid"], ...] = (
            "linear",
            "exp",
            "sigmoid",
        )
        return [
            CurveParams(fam, tail, shape)
            for fam in families
            for tail in (10, 20, 40, 60, 90, 120)
            for shape in (0.5, 1, 2, 5)
        ]

