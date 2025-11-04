from pathlib import Path
import json
from prediction_engine.prediction_engine.artifacts.manager import ArtifactManager

def test_per_symbol_meta_contains_pooled_pointer_if_exists(tmp_path):
    pq = tmp_path / "parquet"; pq.mkdir()
    p = pq / "symbol=RRC/year=2019/month=05/day=07"
    p.mkdir(parents=True, exist_ok=True)
    (p / "part-0.parquet").write_bytes(b"PAR1")

    arte = tmp_path / "artifacts"
    (arte / "pooled").mkdir(parents=True, exist_ok=True)  # pretend pooled exists

    am = ArtifactManager(parquet_root=pq, artifacts_root=arte)
    am.fit_or_load(
        universe=["RRC"],
        start="1999-05-01", end="1999-05-31",
        strategy="per_symbol",
        config_hash_parts={"metric":"mahalanobis"},
        schema_hash_parts={"feature_schema_version": "v1"},
    )

    meta = json.loads((arte / "RRC" / "meta.json").read_text())["payload"]
    assert meta.get("pooled_core_dir") == str(arte / "pooled")
