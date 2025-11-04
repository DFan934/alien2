import json
from pathlib import Path
from prediction_engine.prediction_engine.artifacts.manager import ArtifactManager

def test_calibrator_rejects_low_quality(tmp_path):
    pq = tmp_path / "parquet"; pq.mkdir()
    p = pq / "symbol=RRC/year=1999/month=01/day=05"
    p.mkdir(parents=True, exist_ok=True)
    (p / "part-0.parquet").write_bytes(b"PAR1")

    arte = tmp_path / "artifacts"
    am = ArtifactManager(parquet_root=pq, artifacts_root=arte)

    def fake_pooled(symbols, out_dir, start, end):
        out = Path(out_dir); (out / "calibrators").mkdir(parents=True, exist_ok=True)
        for name in ["scaler.pkl", "pca.pkl", "clusters.pkl", "feature_schema.json"]:
            (out / name).write_text("x", encoding="utf-8")

    # bad metrics: ECE high, monotonic violations, brier above pooled+3%
    def bad_cal(sym, pooled_dir, start, end):
        cal = Path(pooled_dir) / "calibrators"
        (cal / f"{sym}.isotonic.pkl").write_text("x", encoding="utf-8")
        return cal / f"{sym}.isotonic.pkl", {"ece": 0.08, "brier": 0.25, "pooled_brier": 0.20, "mono_adj_pairs": 5}

    am.fit_or_load(
        universe=["RRC"],
        start="1999-01-01", end="1999-01-31",
        strategy="pooled",
        config_hash_parts={"metric":"euclidean"},
        schema_hash_parts={"feature_schema_version":"v1"},
        pooled_builder=fake_pooled,
        calibrator_builder=bad_cal,
    )

    rpt = json.loads((arte / "pooled" / "calibrators" / "calibration_report.json").read_text())
    assert rpt["RRC"]["status"] == "fail"
    assert rpt["RRC"]["ece"] > 0.03
    assert rpt["RRC"]["mono_adj_pairs"] < 8
    assert rpt["RRC"]["brier"] > rpt["RRC"]["pooled_brier"] + rpt["RRC"]["gates"]["brier_delta_max"]
