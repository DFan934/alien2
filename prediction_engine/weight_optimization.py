# ---------------------------------------------------------------------------
# FILE: prediction_engine/weight_optimization.py
# ---------------------------------------------------------------------------
"""Recency‑curve optimiser with train/valid/test split + Optuna fine search.

The optimiser produces **one JSON per market‑regime** so EVEngine can apply
ageing weights dynamically.  A curve is accepted only if
``test_sharpe >= 0.8 * train_sharpe`` to curb over‑fitting.
"""
from __future__ import annotations
#from .weight_optimization import CurveParams
import concurrent.futures as cf
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple, Literal

import numpy as np
import optuna
import pandas as pd
from optuna.exceptions import TrialPruned
from typing import Union



RegimeT = Literal["trend", "range", "volatile", "global"]


# ──────────────────────────────────────────────────────────────────────────────
# In prediction_engine/weight_optimization.py, after the imports:
# ──────────────────────────────────────────────────────────────────────────────

from pathlib import Path
import json

def load_all_regime_curves(artefact_root: str | Path) -> dict[str, CurveParams]:
    """
    Scan artefact_root/regime=<name>/curve_params.json for every regime and
    return a mapping { regime_name: CurveParams(...) }.
    """
    root = Path(artefact_root)
    curves: dict[str, CurveParams] = {}
    for regime_dir in root.glob("regime=*"):
        name = regime_dir.name.split("=", 1)[1]  # “trend”, “range”, …
        params_file = regime_dir / "curve_params.json"
        if not params_file.is_file():
            continue
        data = json.loads(params_file.read_text(encoding="utf-8"))
        # JSON layout: {"params": {family, tail_len, shape}, ...}
        cp = CurveParams(**data["params"])
        curves[name.lower()] = cp
    return curves



@dataclass(slots=True)
class CurveParams:
    family: Literal["linear", "exp", "sigmoid"]
    tail_len: int      # days
    shape: float       # decay / slope
    blend_alpha: float = 0.5 # [0–1] mix synthetic vs cluster EV
    lambda_reg: float  = 1.0 # L2 regularization weight on cluster EV

    def to_dict(self):
        return asdict(self)


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
        rf_feature_importances: np.ndarray | None = None,
        validation_frac: float = 0.2,
        test_frac: float = 0.1,
        artefact_root: Path | str = Path("artifacts/weights"),
    ) -> Dict[str, float]:
        """Return Sharpe metrics and path to persisted JSON artefact."""
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
        #with cf.ProcessPoolExecutor(max_workers=self.n_jobs) as pool:
        #    scores = list(pool.map(lambda p: self._score(p, valid), grid))
        if self.n_jobs > 1 and os.name != "nt":  # Windows pickling quirks
            with cf.ProcessPoolExecutor(max_workers=self.n_jobs) as pool:
                scores = list(pool.map(lambda p: self._score(p, valid), grid))
        else:  # serial fallback (safe / deterministic)
            scores = [self._score(p, valid) for p in grid]

        best_p, best_sh_valid = max(scores, key=lambda t: t[1])

        # ---------- Optuna refine (with pruning) ---------------------
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
        study = optuna.create_study(direction="maximize", pruner=pruner)

        def _obj(trial: optuna.Trial):
            cand = CurveParams(
                family=best_p.family,
                tail_len=trial.suggest_int("tail_len", 5, 120),
                shape=trial.suggest_float("shape", 0.1, 10),
            )
            sharpe = self._score(cand, valid)[1]
            trial.report(sharpe, step=0)
            if trial.should_prune():
                raise TrialPruned()
            return sharpe

        study.optimize(_obj, n_trials=128, show_progress_bar=False)
        bp = best_p.family  # keep the winning family
        best_param_dict = {"family": bp, **study.best_params}
        p_opt = CurveParams(**best_param_dict)  # construct safely

        sharpe_valid = study.best_value
        if sharpe_valid < best_sh_valid:
            p_opt, sharpe_valid = best_p, best_sh_valid

        sharpe_train = self._score(p_opt, train)[1]
        sharpe_test = self._score(p_opt, test)[1]

        # 0.8 over‑fit guard
        # 0.8 over-fit guard – skip for tiny test sets (< 20 obs)
        if len(test) >= 20 and sharpe_test < 0.8 * sharpe_train:
            raise RuntimeError("Curve over-fits – test Sharpe too low vs train")

        # ---------- persist artefacts --------------------------------
        out_dir = Path(artefact_root) / f"regime={regime}"
        out_dir.mkdir(parents=True, exist_ok=True)
        curve_json = out_dir / "curve_params.json"
        curve_json.write_text(
            json.dumps(
                {
                    "params": p_opt.to_dict(),
                    "train_sharpe": sharpe_train,
                    "valid_sharpe": sharpe_valid,
                    "test_sharpe": sharpe_test,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        if rf_feature_importances is not None:
            np.save(out_dir / "rf_feature_weights.npy", rf_feature_importances.astype(np.float32))

        return {
            "file": str(curve_json),
            "train": sharpe_train,
            "valid": sharpe_valid,
            "test": sharpe_test,
        }

    def load_weight_curves(
        artefact_root: Union[str, Path],
    ) -> Dict[RegimeT, CurveParams]:
        """
        Read all regime=<name>/curve_params.json under artefact_root and return
        a dict mapping regime names to CurveParams.
        """
        root = Path(artefact_root)
        curves: dict[RegimeT, CurveParams] = {}
        for sub in root.glob("regime=*"):
            regime = sub.name.split("=", 1)[1]
            js = sub / "curve_params.json"
            if not js.exists():
                continue
            data = json.loads(js.read_text(encoding="utf-8"))
            params = CurveParams(**data["params"])
            curves[regime] = params
        return curves

    # ================= helpers ======================================
    def _score(self, p: CurveParams, pnl: pd.Series) -> Tuple[CurveParams, float]:
        """Sharpe ratio of *pnl* when weighted by curve *p*."""
        # select the last *tail_len* observations so vector lengths match
        slice_len = min(p.tail_len, len(pnl))
        pnl_slice = pnl.iloc[-slice_len:]
        w = self._weights(slice_len, p)
        w /= w.sum()
        ret = float(w @ pnl_slice.values)
        risk = float(np.sqrt(w @ (pnl_slice.values - ret) ** 2)) + 1e-9
        #now also build EV predictions with blend_alpha & lambda_reg:
        #ev_res = ev.evaluate(x, blend_alpha=p.blend_alpha, lambda_reg=p.lambda_reg, …)
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
        tails = (10, 20, 40, 60, 90, 120)
        shapes = (0.5, 1, 2, 5)
        alphas = (0.0, 0.25, 0.5)
        regs = (0.0, 0.1, 1.0)

        return [
            CurveParams(fam, tail, shape,alpha,lam)
            for fam in families
            for tail in tails
            for shape in shapes
            for alpha in alphas
            for lam in regs
        ]

    def list_regimes(artefact_root: Path | str) -> list[str]:
        """
        +    Return the list of regime names for which
        +    artefact_root/regime=<name>/curve_params.json exists.
        +    """
        root = Path(artefact_root)
        return [
            d.name.split("=", 1)[1]
            for d in root.glob("regime=*")
            if (d / "curve_params.json").exists()
        ]


def load_regime_curves(artefact_root: Path | str) -> dict[str, CurveParams]:
    """
    Load all CurveParams from
    artefact_root/regime=<name>/curve_params.json
    and return a mapping { name: CurveParams(...) }.
    """
    root = Path(artefact_root)
    curves: dict[str, CurveParams] = {}
    for d in root.glob("regime=*"):
        f = d / "curve_params.json"
        if not f.exists():
            continue
        payload = json.loads(f.read_text())
        curves[d.name.split("=", 1)[1]] = CurveParams(**payload["params"])
    return curves