from pathlib import Path
import json
from prediction_engine.prediction_engine.artifacts.manager import ArtifactManager

def test_diagnostics_shape_and_values_pooled(tmp_path):
    pq = tmp_path / "parquet"; pq.mkdir()
    for sym in ["RRC","BBY"]:
        p = pq / f"symbol={sym}/year=1999/month=01/day=05"
        p.mkdir(parents=True, exist_ok=True)
        (p / "part-0.parquet").write_bytes(b"PAR1")

    arte = tmp_path / "artifacts"
    am = ArtifactManager(parquet_root=pq, artifacts_root=arte)

    # Minimal pooled builder to create core layout
    def pooled(symbols, out_dir, start, end):
        out = Path(out_dir)
        (out / "calibrators").mkdir(parents=True, exist_ok=True)
        for name in ["scaler.pkl","pca.pkl","clusters.pkl","feature_schema.json"]:
            (out / name).write_text("x", encoding="utf-8")

    # Calibrator builder that writes metrics (so diagnostics can pick them up)
    def cal(sym, pooled_dir, *_):
        caldir = Path(pooled_dir) / "calibrators"
        (caldir / f"{sym}.isotonic.pkl").write_text("x")
        # mimic manager's gating metrics
        return caldir / f"{sym}.isotonic.pkl", {
            "ece": 0.02, "brier": 0.21, "pooled_brier": 0.20, "mono_adj_pairs": 9
        }

    am.fit_or_load(
        universe=["RRC","BBY"],
        start="1999-01-01", end="1999-01-31",
        strategy="pooled",
        config_hash_parts={"metric":"euclidean","k_max":8,"use_isotonic":True},
        schema_hash_parts={"feature_schema_version":"v1"},
        pooled_builder=pooled,
        calibrator_builder=cal,
    )

    dpath = arte / "pooled" / "diagnostics.json"
    assert dpath.exists(), "diagnostics.json must be written for pooled artifacts"

    d = json.loads(dpath.read_text())
    # core keys
    for k in ["distance_family","k_used","feature_schema_version",
              "fallback_rate_by_regime","median_knn_distance_by_regime",
              "calibration_ece_by_symbol","brier_by_symbol"]:
        assert k in d, f"missing diagnostics key {k}"

    # regimes present (no missing keys)
    assert set(d["fallback_rate_by_regime"].keys()) == {"TREND","RANGE","VOL","GLOBAL"}
    assert set(d["median_knn_distance_by_regime"].keys()) == {"TREND","RANGE","VOL","GLOBAL"}

    # calibration maps include both symbols
    assert {"RRC","BBY"} <= set(d["calibration_ece_by_symbol"].keys())
    assert {"RRC","BBY"} <= set(d["brier_by_symbol"].keys())

    # distance echo
    assert d["distance_family"] == "euclidean"
    assert int(d["k_used"]) == 8
