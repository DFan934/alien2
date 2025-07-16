import json
from pathlib import Path

import numpy as np
import pytest

from prediction_engine.ev_engine import EVEngine, _sha1_list
from prediction_engine.weight_optimization import CurveParams
from prediction_engine.market_regime import MarketRegime

def write_cluster_artifacts(base: Path):
    """
    Create minimal cluster artifacts under `base` so that
    EVEngine.from_artifacts(...) will load without error.
    """
    base.mkdir()
    # centers.npy
    centers = np.zeros((1,1), dtype=np.float32)
    np.save(base/"centers.npy", centers)

    # cluster_stats.npz
    stats = {
        "mu": np.array([0.0], dtype=np.float32),
        "var": np.array([1.0], dtype=np.float32),
        "var_down": np.array([1.0], dtype=np.float32),
        "feature_list": np.array(["f1"], dtype=object),
    }
    np.savez(base/"cluster_stats.npz", **stats)

    # meta.json (sha must match _sha1_list(["f1"]))
    meta = {"sha": _sha1_list(["f1"]), "features": ["f1"]}
    (base/"meta.json").write_text(json.dumps(meta))

    # kernel_bandwidth.json
    kernel = {"h": 1.0, "blend_alpha": 0.5}
    (base/"kernel_bandwidth.json").write_text(json.dumps(kernel))

    # outcome_probabilities.json (empty OK)
    (base/"outcome_probabilities.json").write_text(json.dumps({}))

    return base

def test_from_artifacts_picks_up_nightly_curves(tmp_path, monkeypatch):
    # 1. make a cluster artifact set
    cluster_dir = tmp_path/"cluster_artifacts"
    write_cluster_artifacts(cluster_dir)

    # 2. create a nightly-calibrate output under artifacts/weights/regime=trend
    #    and regime=range
    weight_root = tmp_path/"artifacts"/"weights"
    for name, params in {
        "trend": {"family": "linear", "tail_len": 3, "shape": 0.7},
        "range": {"family": "sigmoid","tail_len": 5, "shape": 1.2},
    }.items():
        d = weight_root/f"regime={name}"
        d.mkdir(parents=True)
        (d/"curve_params.json").write_text(
            json.dumps({"params": params})
        )

    # 3. chdir so EVEngine.from_artifacts can find "artifacts/weights"
    monkeypatch.chdir(tmp_path)

    # 4. call the factory
    engine = EVEngine.from_artifacts(cluster_dir)

    # 5. verify that both new curves made it in
    rc = engine.regime_curves
    assert set(rc.keys()) >= {"trend", "range"}

    ct = rc["trend"]
    assert isinstance(ct, CurveParams)
    assert ct.family   == "linear"
    assert ct.tail_len == 3
    assert pytest.approx(ct.shape) == 0.7

    cr = rc["range"]
    assert isinstance(cr, CurveParams)
    assert cr.family   == "sigmoid"
    assert cr.tail_len == 5
    assert pytest.approx(cr.shape) == 1.2
