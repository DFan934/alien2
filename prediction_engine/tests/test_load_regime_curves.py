import json, shutil
from pathlib import Path
import pytest
from prediction_engine.weight_optimization import load_all_regime_curves, CurveParams

def test_load_all_regime_curves(tmp_path):
    # Setup: create two regimes under artifacts
    root = tmp_path / "weights"
    for name, cfg in [("trend", {"family":"exp","tail_len":5,"shape":1.5}),
                      ("range", {"family":"linear","tail_len":3,"shape":2.0})]:
        d = root / f"regime={name}"
        d.mkdir(parents=True)
        data = {"params": cfg, "train_sharpe":1.0, "valid_sharpe":0.8, "test_sharpe":0.9}
        (d/"curve_params.json").write_text(json.dumps(data))

    curves = load_all_regime_curves(root)
    assert set(curves) == {"trend","range"}
    assert isinstance(curves["trend"], CurveParams)
    assert curves["trend"].family == "exp"
    assert curves["range"].tail_len == 3
