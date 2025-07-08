# ---------------------------------------------------------------------------
# scripts/nightly_calibrate.py
# ---------------------------------------------------------------------------
"""Nightly calibration job
~~~~~~~~~~~~~~~~~~~~~~~~~~
1. Loads a parquet slice (last *N* days)
2. Builds/updates path‑clusters and outcome stats
3. Performs kernel‑bandwidth CV for `EVEngine`
4. Drops artefacts into `weights/`

Schedule this via CRON or Windows Task Scheduler to run after market close.
"""
from __future__ import annotations

import argparse
import datetime as dt
import pathlib

import pandas as pd

from prediction_engine.market_regime import RegimeDetector
from prediction_engine.path_cluster_engine import PathClusterEngine
from prediction_engine.weight_optimization import WeightOptimizer


FEATURES = [
    "vwap_dev", "rvol", "ema_21", "roc_60", "atr", "adx",
]


def _load_data(path: pathlib.Path, start: str, end: str) -> pd.DataFrame:
    ds = pd.read_parquet(path, filters=[[("date", ">=", start)], [("date", "<=", end)]])
    return ds


def main(args: argparse.Namespace):
    today = dt.date.today()
    start = (today - dt.timedelta(days=args.lookback)).isoformat()
    end = today.isoformat()

    df = _load_data(pathlib.Path(args.parquet_dir), start, end)
    engine = PathClusterEngine.build(
        df,
        features=FEATURES,
        outcome_col="future_ret",
        n_clusters=args.k,
        cfg={"seed": 42},
        out_dir=args.out_dir,
    )
    print("[nightly_calibrate] saved centres & stats →", args.out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--parquet-dir", required=True)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--lookback", type=int, default=365)
    p.add_argument("--out-dir", default="weights/path_cluster/")
    p.add_argument("--regime-aware", action="store_true")

    main(p.parse_args())