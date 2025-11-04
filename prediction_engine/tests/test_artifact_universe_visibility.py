# NEW FILE: prediction_engine/tests/test_artifact_universe_visibility.py
import json
from pathlib import Path
from prediction_engine.prediction_engine.artifacts.manager import ArtifactManager

def _read_payload(meta_path: Path) -> dict:
    return json.loads(meta_path.read_text())["payload"]

def _seed(parquet_root: Path, symbols):
    for sym in symbols:
        p = parquet_root / f"symbol={sym}/year=2019/month=05/day=07"
        p.mkdir(parents=True, exist_ok=True)
        (p / "part-0.parquet").write_bytes(b"PAR1")

def test_per_symbol_layout_and_universe_hash(tmp_path):
    parquet_root = tmp_path / "parquet"
    arte_root    = tmp_path / "artifacts"
    parquet_root.mkdir(); arte_root.mkdir()
    _seed(parquet_root, ["RRC","BBY"])

    am = ArtifactManager(parquet_root=parquet_root, artifacts_root=arte_root)
    cfg_hash = {"metric": "mahalanobis", "k_max": 32}
    sch_hash = {"feature_schema_version": "v1", "label_horizon_bars": 20}

    # First with single-name universe
    am.fit_or_load(
        universe=["RRC"], start="2019-05-01", end="2019-05-31",
        strategy="per_symbol",
        config_hash_parts=cfg_hash,
        schema_hash_parts=sch_hash
    )
    meta_rrc = arte_root / "RRC" / "meta.json"
    assert meta_rrc.exists()
    u1 = _read_payload(meta_rrc)["universe_hash"]

    # Now with two-name universe—should write BBY and change universe_hash
    am.fit_or_load(
        universe=["RRC","BBY"], start="2019-05-01", end="2019-05-31",
        strategy="per_symbol",
        config_hash_parts=cfg_hash,
        schema_hash_parts=sch_hash
    )
    meta_bby = arte_root / "BBY" / "meta.json"
    assert meta_bby.exists()
    u2 = _read_payload(meta_bby)["universe_hash"]
    assert u1 != u2, "universe_hash must change when the universe changes"

def test_pooled_layout(tmp_path):
    parquet_root = tmp_path / "parquet"
    arte_root    = tmp_path / "artifacts"
    parquet_root.mkdir(); arte_root.mkdir()
    _seed(parquet_root, ["RRC","BBY"])

    am = ArtifactManager(parquet_root=parquet_root, artifacts_root=arte_root)
    cfg_hash = {"metric": "mahalanobis", "k_max": 32}
    sch_hash = {"feature_schema_version": "v1", "label_horizon_bars": 20}

    am.fit_or_load(
        universe=["RRC","BBY"], start="2019-05-01", end="2019-05-31",
        strategy="pooled",
        config_hash_parts=cfg_hash,
        schema_hash_parts=sch_hash
    )
    meta_pooled = arte_root / "pooled" / "meta.json"
    assert meta_pooled.exists(), "pooled/meta.json must exist for pooled strategy"
    payload = _read_payload(meta_pooled)
    assert payload["strategy"] == "pooled"
    assert set(payload.get("symbols", [])) == {"RRC","BBY"}
