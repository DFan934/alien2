# prediction_engine/tests/test_isotonic_calibration.py
import json
from pathlib import Path
from prediction_engine.prediction_engine.artifacts.manager import ArtifactManager

def _seed_core(out: Path):
    (out / "pooled").mkdir(parents=True, exist_ok=True)
    for name in ["scaler.pkl","pca.pkl","clusters.pkl","feature_schema.json"]:
        (out / "pooled" / name).write_text("x")

def test_calibrator_files_exist_and_thresholds_enforced(tmp_path):
    pq = tmp_path / "parquet"; pq.mkdir()
    for sym in ["RRC","BBY"]:
        (pq / f"symbol={sym}/year=1999/month=01/day=05").mkdir(parents=True, exist_ok=True)
        (pq / f"symbol={sym}/year=1999/month=01/day=05/part-0.parquet").write_bytes(b"PAR1")

    arte = tmp_path / "artefacts"
    am = ArtifactManager(parquet_root=pq, artifacts_root=arte)

    def pooled(symbols, out_dir, start, end):
        _seed_core(Path(out_dir))

    # Good metrics → pass
    def good_cal(sym, pooled_dir, start, end):
        cal = Path(pooled_dir) / "calibrators"; cal.mkdir(parents=True, exist_ok=True)
        (cal / f"{sym}.isotonic.pkl").write_text("x")
        return cal / f"{sym}.isotonic.pkl", {"ece": 0.02, "brier": 0.21, "pooled_brier": 0.20, "mono_adj_pairs": 9}

    out = am.fit_or_load(
        universe=["RRC","BBY"], start="1999-01-01", end="1999-01-31",
        strategy="pooled",
        config_hash_parts={"use_isotonic": True},
        schema_hash_parts={"feature_schema_version":"v1"},
        pooled_builder=pooled, calibrator_builder=good_cal
    )
    rpt = json.loads((arte / "pooled" / "calibrators" / "calibration_report.json").read_text())
    for sym in ["RRC","BBY"]:
        assert (arte / "pooled" / "calibrators" / f"{sym}.isotonic.pkl").exists()
        assert rpt[sym]["status"] == "pass"
        assert rpt[sym]["ece"] <= 0.03
        assert rpt[sym]["mono_adj_pairs"] >= 8
        assert rpt[sym]["brier"] <= rpt[sym].get("pooled_brier", 0.0) + rpt[sym]["gates"]["brier_delta_max"]

    # Bad metrics → fail
    def bad_cal(sym, pooled_dir, start, end):
        cal = Path(pooled_dir) / "calibrators"; cal.mkdir(parents=True, exist_ok=True)
        (cal / f"{sym}.isotonic.pkl").write_text("x")
        return cal / f"{sym}.isotonic.pkl", {"ece": 0.08, "brier": 0.27, "pooled_brier": 0.20, "mono_adj_pairs": 5}

    am.fit_or_load(
        universe=["RRC"], start="1999-01-01", end="1999-01-31",
        strategy="pooled",
        config_hash_parts={"use_isotonic": True},
        schema_hash_parts={"feature_schema_version":"v1"},
        pooled_builder=pooled, calibrator_builder=bad_cal
    )
    rpt2 = json.loads((arte / "pooled" / "calibrators" / "calibration_report.json").read_text())
    assert rpt2["RRC"]["status"] == "fail"
