import json
from pathlib import Path
from prediction_engine.prediction_engine.artifacts.manager import ArtifactManager

def test_calibrator_report_thresholds(tmp_path):
    pq = tmp_path / "parquet"; pq.mkdir()
    # seed tiny hive skeleton so manager is happy
    for sym in ["RRC","BBY"]:
        p = pq / f"symbol={sym}/year=1999/month=01/day=05"
        p.mkdir(parents=True, exist_ok=True)
        (p / "part-0.parquet").write_bytes(b"PAR1")

    arte = tmp_path / "artifacts"
    am = ArtifactManager(parquet_root=pq, artifacts_root=arte)

    # fake pooled core (so manager proceeds)
    def fake_pooled(symbols, out_dir, start, end):
        out = Path(out_dir); (out / "calibrators").mkdir(parents=True, exist_ok=True)
        for name in ["scaler.pkl", "pca.pkl", "clusters.pkl", "feature_schema.json"]:
            (out / name).write_text("x", encoding="utf-8")

    # builder that writes good-enough metrics
    def good_cal(sym, pooled_dir, start, end):
        cal = Path(pooled_dir) / "calibrators"
        (cal / f"{sym}.isotonic.pkl").write_text("x", encoding="utf-8")
        return cal / f"{sym}.isotonic.pkl", {"ece": 0.02, "brier": 0.21, "pooled_brier": 0.20, "mono_adj_pairs": 9}

    am.fit_or_load(
        universe=["RRC","BBY"],
        start="1999-01-01", end="1999-01-31",
        strategy="pooled",
        config_hash_parts={"metric":"euclidean"},
        schema_hash_parts={"feature_schema_version":"v1"},
        pooled_builder=fake_pooled,
        calibrator_builder=good_cal,
    )

    rpt = json.loads((arte / "pooled" / "calibrators" / "calibration_report.json").read_text())
    assert set(rpt.keys()) == {"RRC","BBY"}
    for sym in ["RRC","BBY"]:
        assert rpt[sym]["status"] == "pass"
        assert rpt[sym]["ece"] <= 0.03
        assert rpt[sym]["mono_adj_pairs"] >= 8
        # pooled_brier + delta gate
        assert rpt[sym]["brier"] <= (rpt[sym]["gates"]["brier_delta_max"] + rpt[sym].get("pooled_brier", 0.0))
