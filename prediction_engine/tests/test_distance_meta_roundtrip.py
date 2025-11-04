import json
import numpy as np
from pathlib import Path
import pytest
from prediction_engine.ev_engine import EVEngine

def _write_meta(dir_: Path, family: str, rf_w: np.ndarray | None = None, cov: np.ndarray | None = None):
    payload = {
        "fingerprint": {"n_rows": 10, "tmax": "1999-01-02"},
        "config_hash": "abc",
        "schema_hash": "def",
        "universe_hash": "ghi",
        "window": {"start": "1999-01-01", "end": "1999-02-01"},
        "distance": {"family": family, "params": {}},
    }
    if family == "rf_weighted" and rf_w is not None:
        rf_file = dir_ / "rf_feature_weights.npy"
        np.save(rf_file, rf_w.astype(np.float32))
        payload["distance"]["params"]["rf_weights_len"] = int(rf_w.size)
        payload["distance"]["params"]["rf_weights_sha1"] = __import__("hashlib").sha1(
            rf_w.astype(np.float32).tobytes()
        ).hexdigest()[:12]
        payload["distance"]["params"]["rf_weights_path"] = str(rf_file)
    if family == "mahalanobis" and cov is not None:
        inv = np.linalg.inv(cov).astype(np.float32)
        inv_path = dir_ / "centroid_cov_inv.npy"
        np.save(inv_path, inv)
        sha = __import__("hashlib").sha1(inv.tobytes()).hexdigest()[:12]
        payload["distance"]["params"]["cov_inv_sha1"] = sha
        payload["distance"]["params"]["cov_inv_path"] = str(inv_path)

    (dir_ / "meta.json").write_text(json.dumps({"payload": payload}, indent=2))

def _write_minimal_core(dir_: Path, n_feat=4):
    # centers + stats + schema + kernel
    np.save(dir_ / "centers.npy", np.zeros((3, n_feat), dtype=np.float32))
    np.savez(dir_ / "cluster_stats.npz",
             mu=np.zeros(3, dtype=np.float32),
             var=np.ones(3, dtype=np.float32),
             var_down=np.ones(3, dtype=np.float32),
             feature_list=[f"f{i}" for i in range(n_feat)],
             regime=np.array(["GLOBAL","GLOBAL","GLOBAL"]))
    schema = {"features":[f"f{i}" for i in range(n_feat)], "sha": __import__("hashlib").sha1(
        "|".join([f"f{i}" for i in range(n_feat)]).encode()).hexdigest()[:12],
              "scales":[1.0]*n_feat}
    (dir_ / "feature_schema.json").write_text(json.dumps(schema))
    (dir_ / "kernel_bandwidth.json").write_text(json.dumps({"h": 1.0, "blend_alpha": 0.5}))

def test_metric_accept_and_reject(tmp_path):
    arte = tmp_path / "A"; arte.mkdir(parents=True)
    _write_minimal_core(arte, n_feat=4)

    # RF-weighted OK
    rf_w = np.array([0.7,0.2,0.1,0.0], dtype=np.float32)
    _write_meta(arte, family="rf_weighted", rf_w=rf_w)
    eng = EVEngine.from_artifacts(arte, metric="rf_weighted")  # should load fine
    assert eng.metric == "rf_weighted"

    # Mismatch rejected
    with pytest.raises(RuntimeError):
        EVEngine.from_artifacts(arte, metric="euclidean")

    # Mahalanobis OK when inv-cov present and matches sha
    arte2 = tmp_path / "B"; arte2.mkdir()
    _write_minimal_core(arte2, n_feat=3)
    cov = np.eye(3, dtype=np.float32)
    _write_meta(arte2, family="mahalanobis", cov=cov)
    eng2 = EVEngine.from_artifacts(arte2, metric="mahalanobis")
    assert eng2.metric == "mahalanobis"
