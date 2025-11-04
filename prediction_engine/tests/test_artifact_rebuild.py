# NEW TEST FILE
import json, time
from pathlib import Path
from prediction_engine.prediction_engine.artifacts.manager import ArtifactManager

def _read_payload(meta_path: Path) -> dict:
    return json.loads(meta_path.read_text())["payload"]

def test_rebuild_policy(tmp_path):
    # Arrange: temp roots
    parquet_root = tmp_path / "parquet"
    arte_root    = tmp_path / "artifacts"
    parquet_root.mkdir(parents=True, exist_ok=True)
    arte_root.mkdir(parents=True, exist_ok=True)

    # Minimal hive structure with no real data (fingerprint falls back to fs hashing)
    (parquet_root / "symbol=RRC/year=1999/month=01/day=04").mkdir(parents=True, exist_ok=True)
    (parquet_root / "symbol=RRC/year=1999/month=01/day=04/part-0.parquet").write_bytes(b"PAR1")

    am = ArtifactManager(parquet_root=parquet_root, artifacts_root=arte_root)

    cfg_hash = {"metric": "mahalanobis", "k_max": 32}
    sch_hash = {"feature_schema_version": "v1", "label_horizon_bars": 20}

    # 1) First run → builds & writes meta
    am.fit_or_load(universe=["RRC"], start="1999-01-01", end="1999-02-01",
                   strategy="per_symbol",
                   config_hash_parts=cfg_hash,
                   schema_hash_parts=sch_hash)
    meta1 = (arte_root / "RRC" / "meta.json")
    assert meta1.exists(), "meta.json must be written on first build"
    p1 = _read_payload(meta1)

    # 2) Second run with identical inputs → idempotent (no rewrite)
    before = meta1.stat().st_mtime
    time.sleep(0.01)  # timestamp granularity safety
    am.fit_or_load(universe=["RRC"], start="1999-01-01", end="1999-02-01",
                   strategy="per_symbol",
                   config_hash_parts=cfg_hash,
                   schema_hash_parts=sch_hash)
    after = meta1.stat().st_mtime
    assert after == before, "meta.json timestamp unchanged when nothing changed"

    # 3) Change H → should rewrite (rebuild trigger)
    sch_hash2 = {**sch_hash, "label_horizon_bars": 21}
    am.fit_or_load(universe=["RRC"], start="1999-01-01", end="1999-02-01",
                   strategy="per_symbol",
                   config_hash_parts=cfg_hash,
                   schema_hash_parts=sch_hash2)
    after2 = meta1.stat().st_mtime
    assert after2 > after, "meta.json timestamp must update when schema changes"

    # 4) Change metric family → should rewrite
    cfg_hash2 = {**cfg_hash, "metric": "euclidean"}
    am.fit_or_load(universe=["RRC"], start="1999-01-01", end="1999-02-01",
                   strategy="per_symbol",
                   config_hash_parts=cfg_hash2,
                   schema_hash_parts=sch_hash2)
    after3 = meta1.stat().st_mtime
    assert after3 > after2, "meta.json timestamp must update when config changes"

    # Spot-check keys in meta payload
    p2 = _read_payload(meta1)
    for key in ("fingerprint","config_hash","schema_hash","universe_hash","window"):
        assert key in p2, f"payload.{key} must exist"
