# ---------------------------------------------------------------------------
# FILE: scripts/nightly_calibrate.py
# ---------------------------------------------------------------------------
"""Run one WeightOptimizer per market-regime in parallel.

This script should be launched by cron at 02:05 UTC every night **after** the
previous trading day has fully settled and all PnL series are written to
``data/pnl_by_regime/<regime>.csv`` (one column, date index).

Output
------
`artifacts/weights/regime=<name>/curve_params.json` – four JSONs, one per
`MarketRegime` enum.  Existing files are **over-written** – keep last night in
version-control or S3 if needed.
"""
from __future__ import annotations

import concurrent.futures as cf
from pathlib import Path
import pandas as pd
from prediction_engine.market_regime import MarketRegime  # noqa: F401 – only for enum names
from prediction_engine.weight_optimization import WeightOptimizer

_DATA_DIR = Path("data/pnl_by_regime")
_OUT_DIR = Path("artifacts/weights")


def _run_one(regime_name: str) -> tuple[str, dict]:
    csv_path = _DATA_DIR / f"{regime_name}.csv"
    print(f"Running regime: {regime_name} (csv_path={csv_path})")
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    pnl = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if pnl.shape[1] == 1:
        pnl = pnl.iloc[:, 0]  # Convert to Series if only one column
    try:
        res = WeightOptimizer().optimise(pnl, regime=regime_name, artefact_root=_OUT_DIR)
        print(f"Success for {regime_name}")
        return regime_name, res
    except Exception as e:
        print(f"FAILED for {regime_name}: {e}")
        raise




def main() -> None:
    regimes = [r.name.lower() for r in MarketRegime]  # trend, range, volatile, global
    with cf.ProcessPoolExecutor(max_workers=4) as pool:
        for regime, outcome in pool.map(_run_one, regimes):
            print(f"✓ {regime:8s} → test Sharpe {outcome['test']:+.3f}  → {outcome['file']}")


if __name__ == "__main__":
    main()