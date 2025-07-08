# ============================================================================
# FILE: tests/test_weight_opt.py  (NEW)
# ============================================================================
"""Smoke test – weight optimiser writes curve_params.json."""
from pathlib import Path

import numpy as np
import pandas as pd

from prediction_engine.weight_optimization import WeightOptimizer


def test_weight_optimizer_smoke(tmp_path: Path):
    rng = pd.date_range("2024-01-01", periods=90, freq="B")
    pnl = pd.Series(np.random.normal(0.001, 0.01, size=len(rng)), index=rng)
    out_dir = tmp_path / "artifacts/weights"
    res = WeightOptimizer().optimise(pnl, regime="global", artefact_root=out_dir)
    assert Path(res["file"]).exists()