# prediction_engine/tests/test_weight_opt.py

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prediction_engine.weight_optimization import (
    WeightOptimizer,
    CurveParams,
    load_regime_curves,
)


def test_weight_optimizer_smoke(tmp_path: Path):
    """
    Smoke‑test that WeightOptimizer().optimise writes out
    a curve_params.json for a dummy pnl series.
    """
    # create a 90‑day BusinessDay pnl series
    rng = pd.date_range("2024-01-01", periods=90, freq="B")
    pnl = pd.Series(np.random.normal(0.001, 0.01, size=len(rng)), index=rng)

    out_dir = tmp_path / "artifacts" / "weights"
    result = WeightOptimizer().optimise(pnl, regime="global", artefact_root=out_dir)

    # file path is returned and it must exist
    curve_path = Path(result["file"])
    assert curve_path.exists(), f"{curve_path} was not written"


def test_load_regime_curves_single(tmp_path: Path):
    """
    If there’s exactly one regime=<name>/curve_params.json
    we get back a dict with that name→CurveParams.
    """
    root = tmp_path / "weights"
    d = root / "regime=trend"
    d.mkdir(parents=True)

    params = {"family": "exp", "tail_len": 10, "shape": 2.5}
    (d / "curve_params.json").write_text(json.dumps({"params": params}))

    curves = load_regime_curves(root)
    assert set(curves) == {"trend"}

    cp = curves["trend"]
    assert isinstance(cp, CurveParams)
    assert cp.family == "exp"
    assert cp.tail_len == 10
    assert pytest.approx(cp.shape) == 2.5


def test_load_regime_curves_multiple(tmp_path: Path):
    """
    If there are multiple regime=*/curve_params.json directories
    only the ones with a JSON file are returned.
    """
    root = tmp_path / "weights"
    data = {
        "trend":  {"family": "linear",  "tail_len": 5,  "shape": 1.0},
        "range":  {"family": "sigmoid", "tail_len": 8,  "shape": 0.7},
        "global": None,  # no JSON here, should be skipped
    }

    for name, p in data.items():
        d = root / f"regime={name}"
        d.mkdir(parents=True)
        if p is not None:
            (d / "curve_params.json").write_text(json.dumps({"params": p}))

    curves = load_regime_curves(root)
    # only trend and range should survive
    assert set(curves) == {"trend", "range"}

    # spot‑check one of them
    cp_range = curves["range"]
    assert isinstance(cp_range, CurveParams)
    assert cp_range.family == "sigmoid"
    assert cp_range.tail_len == 8
    assert pytest.approx(cp_range.shape) == 0.7
