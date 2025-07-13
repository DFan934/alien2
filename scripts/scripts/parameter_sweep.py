#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# Grid-search over gap-pct, RVOL thresh, PCA variance in a *single* run.
# ---------------------------------------------------------------------------
from __future__ import annotations
import itertools, logging, math, tempfile
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np

from scripts.scripts import run_backtest
from scripts.scripts.run_backtest import run as run_bt

# ────────────────────────────────────────────────────────────────────────
# Define your parameter grid  ⬇︎  (edit here)
# ────────────────────────────────────────────────────────────────────────
GRID = {
    "gap_pct":  [0.02, 0.03, 0.05],
    "rvol":     [1.5, 2.0, 3.0],
    "pca_var":  [0.90, 0.95, 0.98],   # CoreFeaturePipeline.settings.pca_variance
}
# Output CSV
CSV_OUT = Path("sweep_results.csv")
# Path to a *single* raw CSV you already back-test with
RAW_CSV  = "raw_data/RRC.csv"
SYMBOL   = "RRC"
START    = "1998-08-26"
END      = "1998-09-26"
# ────────────────────────────────────────────────────────────────────────


def _sharpe(returns: np.ndarray, rf: float = 0.0) -> float:
    """Daily Sharpe of equity curve (assumes 1 return per bar)."""
    if returns.std() == 0:
        return 0.0
    return (returns.mean() - rf) / returns.std() * math.sqrt(390)  # 390 bars/day


def _single_run(params: Dict[str, float]) -> Dict[str, float]:
    """Run *one* back-test with the supplied params & return metrics."""
    # --- 1. temp artefact dir so runs don’t collide ------------
    with tempfile.TemporaryDirectory() as tmp_art:
        cfg = run_backtest.CONFIG.copy()
        cfg.update({
            "csv": RAW_CSV,
            "symbol": SYMBOL,
            "start": START,
            "end":   END,
            "artefacts": tmp_art,            # build fresh each run
            "out": f"/tmp/{SYMBOL}_signals.csv",
        })
        # weave in sweep params
        cfg["gap_pct"]        = params["gap_pct"]
        cfg["rvol"]           = params["rvol"]
        # override CoreFeaturePipeline variance dynamically
        from feature_engineering.config import settings
        settings.pca_variance = params["pca_var"]

        # run – back-test writes signals CSV, prints stats
        run_bt(cfg)

        # read equity curve to compute metrics
        df_sig = pd.read_csv(cfg["out"])
        ret = df_sig["equity"].pct_change().fillna(0).to_numpy()

        metrics = {
            "total_return_%": (df_sig["equity"].iloc[-1] /
                               df_sig["equity"].iloc[0] - 1) * 100,
            "sharpe": _sharpe(ret),
            "bars": len(df_sig),
        }
        return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    rows: List[Dict[str, float]] = []
    for combo in itertools.product(*GRID.values()):
        params = dict(zip(GRID.keys(), combo))
        logging.info("Running %s", params)
        metrics = _single_run(params)
        rows.append({**params, **metrics})
        logging.info("→ return %.2f %% | Sharpe %.2f",
                     metrics["total_return_%"], metrics["sharpe"])

    out_df = pd.DataFrame(rows)
    out_df.to_csv(CSV_OUT, index=False)
    logging.info("Sweep finished. Results → %s", CSV_OUT)


if __name__ == "__main__":
    main()
