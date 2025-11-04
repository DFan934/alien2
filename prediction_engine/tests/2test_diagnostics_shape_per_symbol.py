from pathlib import Path
import json
from prediction_engine.prediction_engine.artifacts.manager import ArtifactManager

def test_diagnostics_written_per_symbol(tmp_path):
    pq = tmp_path / "pq"; pq.mkdir()
    p = pq / "symbol=RRC/year=1999/month=01/day=05"
    p.mkdir(parents=True, exist_ok=True)
    (p / "part-0.parquet").write_bytes(b"PAR1")

    arte = tmp_path / "arte"
    am = ArtifactManager(parquet_root=pq, artifacts_root=arte)

    # per-symbol builder not required; manager writes meta/diagnostics even without heavy build
    out = am.fit_or_load(
        universe=["RRC"],
        start="1999-01-01", end="1999-01-31",
        strategy="per_symbol",
        config_hash_parts={"metric":"rf_weighted","k_max":32},
        schema_hash_parts={"feature_schema_version":"v2"},
    )

    sym_dir = arte / "RRC"
    dpath = sym_dir / "diagnostics.json"
    assert dpath.exists(), "diagnostics.json must exist for per_symbol artifacts"
    d = json.loads(dpath.read_text())
    for k in ["distance_family","k_used","feature_schema_version",
              "fallback_rate_by_regime","median_knn_distance_by_regime",
              "calibration_ece_by_symbol","brier_by_symbol"]:
        assert k in d
    assert set(d["fallback_rate_by_regime"].keys()) == {"TREND","RANGE","VOL","GLOBAL"}
    assert set(d["median_knn_distance_by_regime"].keys()) == {"TREND","RANGE","VOL","GLOBAL"}
    assert d["distance_family"] == "rf_weighted"
    assert int(d["k_used"]) == 32
