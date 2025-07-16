import json
from pathlib import Path

import numpy as np
import pytest

from prediction_engine.ev_engine import EVEngine, _sha1_list
from prediction_engine.weight_optimization import CurveParams, WeightOptimizer
from prediction_engine.market_regime import MarketRegime


def _make_cluster_artifacts(root: Path) -> None:
    """
    Create the minimal files under `root` so that
    EVEngine.from_artifacts(root) will load without errors.
    """
    # 1 cluster, 2 features
    centers = np.zeros((1, 2), dtype=np.float32)
    np.save(root / "centers.npy", centers)

    # cluster_stats.npz needs mu, var, var_down, feature_list
    feature_list = np.array(["f1", "f2"], dtype=object)
    mu = np.array([0.1], dtype=np.float32)
    var = np.array([0.2], dtype=np.float32)
    var_down = np.array([0.3], dtype=np.float32)
    np.savez(
        root / "cluster_stats.npz",
        mu=mu,
        var=var,
        var_down=var_down,
        feature_list=feature_list,
    )

    # meta.json with matching SHA of feature_list
    sha = _sha1_list(feature_list.tolist())
    meta = {"sha": sha, "features": feature_list.tolist()}
    (root / "meta.json").write_text(json.dumps(meta))

    # kernel bandwidth
    (root / "kernel_bandwidth.json").write_text(
        json.dumps({"h": 1.0, "blend_alpha": 0.5})
    )

    # (optional) empty outcome probabilities
    (root / "outcome_probabilities.json").write_text(json.dumps({}))


def test_ev_engine_loads_nightly_curves(tmp_path: Path, monkeypatch):
    # make tmp_path our CWD so that Path("artifacts/weights") resolves there
    monkeypatch.chdir(tmp_path)

    # 1) Create cluster artifacts folder
    cluster_dir = tmp_path / "cluster_artifacts"
    cluster_dir.mkdir()
    _make_cluster_artifacts(cluster_dir)

    # 2) Create nightly_calibrate output under artifacts/weights
    weight_root = tmp_path / "artifacts" / "weights"
    regime = "trend"
    out_dir = weight_root / f"regime={regime}"
    out_dir.mkdir(parents=True)
    params = {"family": "linear", "tail_len": 3, "shape": 4.2}
    (out_dir / "curve_params.json").write_text(json.dumps({"params": params}))

    # 3) Load EVEngine, which should merge in that regime curve
    engine = EVEngine.from_artifacts(cluster_dir, metric="euclidean")

    # 4) Assert that our nightly curve appears
    assert regime in engine.regime_curves, "EVEngine did not pick up nightly curve"
    cp: CurveParams = engine.regime_curves[regime]
    assert isinstance(cp, CurveParams)
    assert cp.family == "linear"
    assert cp.tail_len == 3
    assert pytest.approx(cp.shape) == 4.2

    # 5) And ensure _get_recency_weights now uses it
    rec_w = engine._get_recency_weights(MarketRegime.TREND, 3)
    expected = WeightOptimizer._weights(3, cp)
    assert np.allclose(rec_w, expected), f"{rec_w} != {expected}"


def test_ev_engine_no_weights_folder(tmp_path: Path, monkeypatch):
    """
    If there is no `artifacts/weights` tree, from_artifacts
    should still succeed and just have an empty regime_curves.
    """
    monkeypatch.chdir(tmp_path)

    cluster_dir = tmp_path / "cluster_only"
    cluster_dir.mkdir()
    _make_cluster_artifacts(cluster_dir)

    # no `artifacts/weights` at all
    engine = EVEngine.from_artifacts(cluster_dir, metric="euclidean")
    # should load the cluster‐side regime_curves (none), and
    # not crash when looking for nightly curves
    assert isinstance(engine.regime_curves, dict)
    assert engine.regime_curves == {}
