# NEW FILE: prediction_engine/tests/test_artifact_rebuild_regime_settings.py
import json, time
from pathlib import Path
from prediction_engine.prediction_engine.artifacts.manager import ArtifactManager

def _read_payload(meta_path: Path) -> dict:
    return json.loads(meta_path.read_text())["payload"]

def _seed_parquet(parquet_root: Path, sym: str = "RRC"):
    p = parquet_root / f"symbol={sym}/year=2014/month=06/day=02"
    p.mkdir(parents=True, exist_ok=True)
    (p / "part-0.parquet").write_bytes(b"PAR1")

def test_regime_settings_change_triggers_rebuild(tmp_path):
    parquet_root = tmp_path / "parquet"
    arte_root    = tmp_path / "artifacts"
    parquet_root.mkdir(); arte_root.mkdir()
    _seed_parquet(parquet_root, "RRC")

    am = ArtifactManager(parquet_root=parquet_root, artifacts_root=arte_root)

    cfg_hash = {"metric": "mahalanobis", "k_max": 32}
    sch_hash = {
        "feature_schema_version": "v1",
        "feature_schema_list": ["vwap_z","rvol_20","ema9_slope"],
        "label_horizon_bars": 20,
        "distance_family": "mahalanobis",
        "regime_settings": {"modes": ["TREND","RANGE","VOL"]},
        "gating_flags": {"p_gate_quantile": 0.55, "full_p_quantile": 0.65, "sign_check": False},
    }

    am.fit_or_load(
        universe=["RRC"], start="2014-06-01", end="2014-07-01",
        strategy="per_symbol",
        config_hash_parts=cfg_hash,
        schema_hash_parts=sch_hash
    )
    meta = arte_root / "RRC" / "meta.json"
    t1 = meta.stat().st_mtime
    payload1 = _read_payload(meta)

    # Change only regime settings (drop VOL)
    time.sleep(0.01)
    sch_hash2 = dict(sch_hash)
    sch_hash2["regime_settings"] = {"modes": ["TREND","RANGE"]}

    am.fit_or_load(
        universe=["RRC"], start="2014-06-01", end="2014-07-01",
        strategy="per_symbol",
        config_hash_parts=cfg_hash,
        schema_hash_parts=sch_hash2
    )
    t2 = meta.stat().st_mtime
    payload2 = _read_payload(meta)
    assert t2 > t1, "changing regime settings must trigger rebuild"
    assert payload1["schema_hash"] != payload2["schema_hash"]
