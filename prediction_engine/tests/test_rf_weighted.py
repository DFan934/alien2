import json

import numpy as np
import pytest
from pathlib import Path

from prediction_engine.distance_calculator import DistanceCalculator
from prediction_engine.ev_engine import EVEngine

def test_distance_calculator_auto_loads_rf_weights(tmp_path: Path):
    # 1) write a simple centers.npy
    centers = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    np.save(tmp_path / "centers.npy", centers)

    # 2) write matching rf_feature_weights.npy
    rf_w = np.array([2.0, 0.5], dtype=np.float32)
    np.save(tmp_path / "rf_feature_weights.npy", rf_w)

    # 3) load via from_artifacts
    dc = DistanceCalculator.from_artifacts(
        ref_path=tmp_path / "centers.npy",
        metric="rf_weighted",
    )
    # it should pick up the weight vector
    assert hasattr(dc, "_rf_w")
    assert dc._rf_w.shape == (2,)
    assert np.allclose(dc._rf_w, rf_w)

    # 4) check that distances use these weights:
    #    for x=[1,0], dist² to c0=(1-0)²·2 + (0-0)²·0.5 = 2.0
    #                     to c1=(1-1)²·2 + (0-1)²·0.5 = 0.5
    x = np.array([1.0, 0.0], dtype=np.float32)
    dists, idxs = dc.top_k(x, 2)
    # distances should be sorted ascending [0.5, 2.0]
    assert np.allclose(dists, np.array([0.5, 2.0], dtype=np.float32))
    # and the nearest‑first index should be the second center (index 1)
    assert idxs[0] == 1

def test_ev_engine_factory_with_rf_weights(tmp_path: Path, monkeypatch):
    # make a minimal cluster_artifacts folder:
    cluster_dir = tmp_path / "cluster_artifacts"
    cluster_dir.mkdir()

    # (a) centers.npy
    centers = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    np.save(cluster_dir / "centers.npy", centers)

    # (b) cluster_stats.npz with mu, var, var_down, feature_list
    feature_list = np.array(["f1", "f2"], dtype=object)
    mu        = np.array([0.1, -0.1], dtype=np.float32)
    var       = np.ones(2, dtype=np.float32)
    var_down  = np.ones(2, dtype=np.float32)
    np.savez(
        cluster_dir / "cluster_stats.npz",
        mu=mu,
        var=var,
        var_down=var_down,
        feature_list=feature_list,
    )

    # (c) meta.json matching the feature_list SHA
    from prediction_engine.ev_engine import _sha1_list
    sha = _sha1_list(feature_list.tolist())
    meta = {"sha": sha, "features": feature_list.tolist()}
    (cluster_dir / "meta.json").write_text(json.dumps(meta))

    # (d) kernel_bandwidth.json
    (cluster_dir / "kernel_bandwidth.json").write_text(
        json.dumps({"h": 1.0, "blend_alpha": 0.5})
    )

    # (e) empty outcome_probabilities.json
    (cluster_dir / "outcome_probabilities.json").write_text(json.dumps({}))

    # (f) RF weights next to centers
    rf_w = np.array([3.0, 1.0], dtype=np.float32)
    np.save(cluster_dir / "rf_feature_weights.npy", rf_w)

    # switch cwd so EVEngine.from_artifacts sees no stray artifacts/weights
    monkeypatch.chdir(tmp_path)

    # now load EVEngine with rf_weighted metric
    engine = EVEngine.from_artifacts(cluster_dir, metric="rf_weighted")

    # its DistanceCalculator should have picked up the same rf weights
    dc = engine._dist
    assert hasattr(dc, "_rf_w")
    assert np.allclose(dc._rf_w, rf_w)

    # And evaluate() should choose the nearest cluster under that weighted norm.
    # For x=[1,0], weighted‐dist to cluster0=(1^2·3 + 0^2·1)=3
    #                         to cluster1=(0^2·3 + (1-0)^2·1)=1 → picks index 1
    x = np.array([1.0, 0.0], dtype=np.float32)
    res = engine.evaluate(x, regime=None)
    assert res.cluster_id == 1
