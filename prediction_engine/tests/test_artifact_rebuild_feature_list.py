# NEW FILE: prediction_engine/tests/test_artifact_rebuild_feature_list.py
import json, time
from pathlib import Path
from prediction_engine.prediction_engine.artifacts.manager import ArtifactManager

def _read_payload(meta_path: Path) -> dict:
    return json.loads(meta_path.read_text())["payload"]

def _seed_parquet(parquet_root: Path, sym: str = "RRC"):
    p = parquet_root / f"symbol={sym}/year=1999/month=01/day=04"
    p.mkdir(parents=True, exist_ok=True)
    (p / "part-0.parquet").write_bytes(b"PAR1")

def test_feature_list_change_triggers_rebuild(tmp_path):
    parquet_root = tmp_path / "parquet"
    arte_root    = tmp_path / "artifacts"
    parquet_root.mkdir(); arte_root.mkdir()
    _seed_parquet(parquet_root, "RRC")

    am = ArtifactManager(parquet_root=parquet_root, artifacts_root=arte_root)

    cfg_hash = {"metric": "mahalanobis", "k_max": 32}
    # First run with schema v1 and a small list
    sch_hash_v1 = {
        "feature_schema_version": "v1",
        "feature_schema_list": ["vwap_z", "rvol_20", "ema9_slope"],
        "label_horizon_bars": 20,
        "distance_family": "mahalanobis",
        "regime_settings": {"modes": ["TREND","RANGE","VOL"]},
        "gating_flags": {"p_gate_quantile": 0.55, "full_p_quantile": 0.65, "sign_check": False},
    }

    am.fit_or_load(
        universe=["RRC"], start="1999-01-01", end="1999-02-01",
        strategy="per_symbol",
        config_hash_parts=cfg_hash,
        schema_hash_parts=sch_hash_v1
    )
    meta = arte_root / "RRC" / "meta.json"
    assert meta.exists()
    payload1 = _read_payload(meta)
    t1 = meta.stat().st_mtime

    # Idempotence: run again, unchanged → same timestamp
    time.sleep(0.01)
    am.fit_or_load(
        universe=["RRC"], start="1999-01-01", end="1999-02-01",
        strategy="per_symbol",
        config_hash_parts=cfg_hash,
        schema_hash_parts=sch_hash_v1
    )
    t2 = meta.stat().st_mtime
    assert t2 == t1

    # Change ONLY feature schema (version and/or list) → must rebuild
    sch_hash_v2 = dict(sch_hash_v1)
    sch_hash_v2["feature_schema_version"] = "v2"
    sch_hash_v2["feature_schema_list"] = sch_hash_v1["feature_schema_list"] + ["adx_14"]

    am.fit_or_load(
        universe=["RRC"], start="1999-01-01", end="1999-02-01",
        strategy="per_symbol",
        config_hash_parts=cfg_hash,
        schema_hash_parts=sch_hash_v2
    )
    t3 = meta.stat().st_mtime
    assert t3 > t2, "schema/list change must trigger rebuild"
    payload2 = _read_payload(meta)
    assert payload1["schema_hash"] != payload2["schema_hash"], "schema_hash must change when feature schema changes"
