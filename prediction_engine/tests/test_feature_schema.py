import json, numpy as np
from pathlib import Path
from prediction_engine.path_cluster_engine import PathClusterEngine
from prediction_engine.ev_engine import EVEngine


def test_schema_guard(tmp_path: Path):
    """Tamper with schema; expect EVEngine to raise."""
    #  — build fake artefacts —
    X = np.random.randn(100, 4).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    feats = ["a", "b", "c", "d"]
    PathClusterEngine.build(
        X, y, None, feature_names=feats, n_clusters=4, out_dir=tmp_path
    )

    #  — tamper —
    meta_path = tmp_path / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["sha"] = "hacked_sha_value"  # Corrupt the hash
    meta_path.write_text(json.dumps(meta))

    #  — expect failure —
    try:
        EVEngine.from_artifacts(tmp_path)
    except RuntimeError as e:
        assert "schema drift" in str(e).lower()
    else:
        raise AssertionError("EVEngine accepted wrong schema!")