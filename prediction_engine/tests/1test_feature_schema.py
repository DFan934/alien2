# prediction_engine/tests/test_feature_schema.py
from pathlib import Path
import json
from prediction_engine.prediction_engine.artifacts.manager import ArtifactManager

def _read_meta(p: Path) -> dict:
    return json.loads((p / "pooled" / "meta.json").read_text())

def _seed_core(out: Path):
    (out / "pooled").mkdir(parents=True, exist_ok=True)
    # minimal core so manager proceeds to meta write
    for name in ["scaler.pkl", "pca.pkl", "clusters.pkl", "feature_schema.json"]:
        (out / "pooled" / name).write_text("x", encoding="utf-8")

def test_schema_hash_change_forces_rebuild(tmp_path):
    pq = tmp_path / "parquet"; pq.mkdir()
    (pq / "symbol=RRC/year=1999/month=01/day=05").mkdir(parents=True, exist_ok=True)
    (pq / "symbol=RRC/year=1999/month=01/day=05/part-0.parquet").write_bytes(b"PAR1")

    arte = tmp_path / "artifacts"
    am = ArtifactManager(parquet_root=pq, artifacts_root=arte)

    # Fake pooled builder (writes core files)
    def fake_pooled(symbols, out_dir, start, end):
        _seed_core(Path(out_dir))

    # First build
    am.fit_or_load(
        universe=["RRC"], start="1999-01-01", end="1999-01-31",
        strategy="pooled",
        config_hash_parts={"metric":"euclidean","k_max":16},
        schema_hash_parts={"feature_schema_version": "v1"},
        pooled_builder=fake_pooled
    )
    meta1 = _read_meta(arte)
    h1 = meta1["payload"]["schema_hash"]

    # Second run with unchanged schema → no change
    am.fit_or_load(
        universe=["RRC"], start="1999-01-01", end="1999-01-31",
        strategy="pooled",
        config_hash_parts={"metric":"euclidean","k_max":16},
        schema_hash_parts={"feature_schema_version": "v1"},
        pooled_builder=fake_pooled
    )
    meta2 = _read_meta(arte)
    h2 = meta2["payload"]["schema_hash"]
    assert h2 == h1, "Idempotent run should not alter schema_hash"

    # Third run with changed schema → must change hash and mark rebuild trigger
    am.fit_or_load(
        universe=["RRC"], start="1999-01-01", end="1999-01-31",
        strategy="pooled",
        config_hash_parts={"metric":"euclidean","k_max":16},
        schema_hash_parts={"feature_schema_version": "v2"},  # <- changed
        pooled_builder=fake_pooled
    )
    meta3 = _read_meta(arte)
    h3 = meta3["payload"]["schema_hash"]
    assert h3 != h2, "Changing feature schema must force rebuild (hash change)"
