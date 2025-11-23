# prediction_engine/tests/test_fold_artifacts_rebuild.py
import json
from pathlib import Path

from prediction_engine.prediction_engine.artifacts.manager import ArtifactManager

def _mk_stub_hive(root: Path, sym: str, y: int, m: int, d: int) -> None:
    p = root / f"symbol={sym}/year={y}/month={m:02d}/day={d:02d}"
    p.mkdir(parents=True, exist_ok=True)
    # We don't need valid parquet; manager's fallback hashes file stats for test writes
    (p / "part-0.parquet").write_bytes(b"PAR1")

def test_build_fold_artifacts_rebuild_on_config(tmp_path: Path):
    # --- Arrange: tiny hive with 2 syms, a few days
    pq = tmp_path / "parquet"
    _mk_stub_hive(pq, "RRC", 1999, 1, 5)
    _mk_stub_hive(pq, "BBY", 1999, 1, 5)

    arte = tmp_path / "artifacts"
    fold_dir = arte / "folds" / "fold_000"

    # --- Track builder calls
    calls = {"pooled": 0, "calib": 0}
    def pooled_builder(symbols, out_dir: Path, start, end):
        calls["pooled"] += 1
        (out_dir / "scaler.pkl").write_text("x")
        (out_dir / "pca.pkl").write_text("x")
        (out_dir / "clusters.pkl").write_text("x")
        (out_dir / "feature_schema.json").write_text("{}")

    def calibrator_builder(sym, pooled_dir: Path, start, end):
        calls["calib"] += 1
        cp = pooled_dir / "calibrators" / f"{sym}.isotonic.pkl"
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text("cal")
        # Minimal metrics contract
        return str(cp), {"ece": 0.02, "brier": 0.18, "mono_adj_pairs": 8, "pooled_brier": 0.18}

    am = ArtifactManager(parquet_root=pq, artifacts_root=arte)

    # --- First build
    out = am.build_fold_artifacts(
        universe=["RRC", "BBY"],
        train_start="1999-01-01",
        train_end="1999-02-01",
        fold_dir=fold_dir,
        strategy="pooled",
        fold_id=0,
        config_hash_parts={"metric": "euclidean", "k_max": 32, "H": 5},
        schema_hash_parts={"feature_schema_version": "v1"},
        pooled_builder=pooled_builder,
        calibrator_builder=calibrator_builder,
    )
    pooled_dir = out["__pooled__"]
    meta_path = pooled_dir / "meta.json"
    assert meta_path.exists()
    m1 = json.loads(meta_path.read_text())["payload"]

    # --- Second call with identical config: no rebuild
    calls_before = calls.copy()
    am.build_fold_artifacts(
        universe=["RRC", "BBY"],
        train_start="1999-01-01",
        train_end="1999-02-01",
        fold_dir=fold_dir,
        strategy="pooled",
        fold_id=0,
        config_hash_parts={"metric": "euclidean", "k_max": 32, "H": 5},
        schema_hash_parts={"feature_schema_version": "v1"},
        pooled_builder=pooled_builder,
        calibrator_builder=calibrator_builder,
    )
    assert calls == calls_before, "Builders should not run when config/data unchanged"

    # --- Third call with a config change (e.g., horizon H): forces rebuild
    am.build_fold_artifacts(
        universe=["RRC", "BBY"],
        train_start="1999-01-01",
        train_end="1999-02-01",
        fold_dir=fold_dir,
        strategy="pooled",
        fold_id=0,
        config_hash_parts={"metric": "euclidean", "k_max": 32, "H": 10},  # changed
        schema_hash_parts={"feature_schema_version": "v1"},
        pooled_builder=pooled_builder,
        calibrator_builder=calibrator_builder,
    )
    assert calls["pooled"] == calls_before["pooled"] + 1
    assert calls["calib"] >= calls_before["calib"] + 2  # one per symbol

    # --- Meta changed
    m2 = json.loads(meta_path.read_text())["payload"]
    assert m2["config_hash"] != m1["config_hash"]
    assert m2["train_window"]["start"] == "1999-01-01"
    assert m2["train_window"]["end"] == "1999-02-01"
    assert "schema_hash" in m2 and "universe_hash" in m2




def test_rebuild_on_data_change(tmp_path):
    from prediction_engine.prediction_engine.artifacts.manager import ArtifactManager
    pq = tmp_path / "parquet"; pq.mkdir()
    # initial day
    d1 = pq / "symbol=RRC/year=1999/month=01/day=05"; d1.mkdir(parents=True, exist_ok=True)
    (d1 / "part-0.parquet").write_bytes(b"PAR1")

    arte = tmp_path / "artifacts"
    fold_dir = arte / "folds" / "fold_000"
    calls = {"pooled":0}
    def pooled_builder(symbols, out_dir, *_):
        calls["pooled"] += 1
        (out_dir / "scaler.pkl").write_text("x")
        (out_dir / "pca.pkl").write_text("x")
        (out_dir / "clusters.pkl").write_text("x")
        (out_dir / "feature_schema.json").write_text("{}")

    am = ArtifactManager(parquet_root=pq, artifacts_root=arte)
    # first build
    am.build_fold_artifacts(universe=["RRC"], train_start="1999-01-01", train_end="1999-02-01",
                            fold_dir=fold_dir, strategy="pooled", fold_id=0,
                            config_hash_parts={"H":5}, schema_hash_parts={"ver":"v1"},
                            pooled_builder=pooled_builder)

    # no-op build (no change)
    before = calls["pooled"]
    am.build_fold_artifacts(universe=["RRC"], train_start="1999-01-01", train_end="1999-02-01",
                            fold_dir=fold_dir, strategy="pooled", fold_id=0,
                            config_hash_parts={"H":5}, schema_hash_parts={"ver":"v1"},
                            pooled_builder=pooled_builder)
    assert calls["pooled"] == before

    # add a new day → fingerprint should change → rebuild
    d2 = pq / "symbol=RRC/year=1999/month=01/day=06"; d2.mkdir(parents=True, exist_ok=True)
    (d2 / "part-0.parquet").write_bytes(b"PAR2")
    am.build_fold_artifacts(universe=["RRC"], train_start="1999-01-01", train_end="1999-02-01",
                            fold_dir=fold_dir, strategy="pooled", fold_id=0,
                            config_hash_parts={"H":5}, schema_hash_parts={"ver":"v1"},
                            pooled_builder=pooled_builder)
    assert calls["pooled"] == before + 1





def test_per_symbol_layout_and_meta(tmp_path):
    from prediction_engine.prediction_engine.artifacts.manager import ArtifactManager
    pq = tmp_path / "parquet"
    for s in ["RRC","BBY"]:
        p = pq / f"symbol={s}/year=1999/month=01/day=05"; p.mkdir(parents=True, exist_ok=True)
        (p / "part-0.parquet").write_bytes(b"PAR1")
    arte = tmp_path / "artifacts"
    fold_dir = arte / "folds" / "fold_001"
    calls = {"sym": []}
    def per_symbol_builder(sym, out_dir, *_):
        calls["sym"].append(sym)
        (out_dir / "feature_schema.json").write_text("{}")
        (out_dir / "clusters.pkl").write_text("x")
    am = ArtifactManager(parquet_root=pq, artifacts_root=arte)
    out = am.build_fold_artifacts(universe=["RRC","BBY"], train_start="1999-01-01", train_end="1999-02-01",
                                  fold_dir=fold_dir, strategy="per_symbol", fold_id=1,
                                  config_hash_parts={"H":5}, schema_hash_parts={"ver":"v1"},
                                  per_symbol_builder=per_symbol_builder)
    assert (fold_dir / "RRC" / "meta.json").exists()
    assert (fold_dir / "BBY" / "meta.json").exists()
    assert set(calls["sym"]) == {"RRC","BBY"}





def test_meta_has_required_fields(tmp_path):
    from prediction_engine.prediction_engine.artifacts.manager import ArtifactManager
    pq = tmp_path / "parquet"; (pq / "symbol=RRC/year=1999/month=01/day=05").mkdir(parents=True, exist_ok=True)
    (pq / "symbol=RRC/year=1999/month=01/day=05/part-0.parquet").write_bytes(b"PAR1")
    arte = tmp_path / "artifacts"; fold_dir = arte / "folds" / "fold_002"
    def pooled_builder(symbols, out_dir, *_):
        (out_dir / "scaler.pkl").write_text("x"); (out_dir / "pca.pkl").write_text("x")
        (out_dir / "clusters.pkl").write_text("x"); (out_dir / "feature_schema.json").write_text("{}")
    am = ArtifactManager(parquet_root=pq, artifacts_root=arte)
    out = am.build_fold_artifacts(universe=["RRC"], train_start="1999-01-01", train_end="1999-02-01",
                                  fold_dir=fold_dir, strategy="pooled", fold_id=2,
                                  config_hash_parts={"H":5,"metric":"euclidean"}, schema_hash_parts={"ver":"v1"},
                                  pooled_builder=pooled_builder)
    meta = json.loads((out["__pooled__"] / "meta.json").read_text())["payload"]
    for k in ["fold_id","train_window","fingerprint","config_hash","schema_hash","universe_hash"]:
        assert k in meta
