import json
from pathlib import Path
import pytest

from prediction_engine.prediction_engine.artifacts.manager import ArtifactManager

def test_pooled_core_and_calibrators_created(tmp_path, monkeypatch):
    # Arrange: seed a tiny hive
    pq = tmp_path / "parquet"; pq.mkdir()
    for sym in ["RRC","BBY"]:
        p = pq / f"symbol={sym}/year=2019/month=05/day=07"
        p.mkdir(parents=True, exist_ok=True)
        (p / "part-0.parquet").write_bytes(b"PAR1")

    arte = tmp_path / "artifacts"
    am = ArtifactManager(parquet_root=pq, artifacts_root=arte)

    # Monkeypatch builders to create the required files quickly
    def fake_pooled_builder(symbols, out_dir, start, end):
        out = Path(out_dir); (out / "calibrators").mkdir(parents=True, exist_ok=True)
        for name in ["scaler.pkl", "pca.pkl", "clusters.pkl", "feature_schema.json"]:
            (out / name).write_text("x", encoding="utf-8")

    def fake_calibrator_builder(sym, pooled_dir, start, end):
        cal = Path(pooled_dir) / "calibrators"
        (cal / f"{sym}.isotonic.pkl").write_text("x", encoding="utf-8")
        rpt = cal / "calibration_report.json"
        try:
            d = json.loads(rpt.read_text())
        except Exception:
            d = {}
        d[sym] = {"roc_auc": 0.60, "brier": 0.24, "ece": 0.02, "n": 100}
        rpt.write_text(json.dumps(d, indent=2), encoding="utf-8")

    # Act
    out = am.fit_or_load(
        universe=["RRC","BBY"],
        start="1999-05-01", end="1999-05-31",
        strategy="pooled",
        config_hash_parts={"metric":"mahalanobis"},
        schema_hash_parts={"feature_schema_version": "v1"},
        pooled_builder=fake_pooled_builder,
        calibrator_builder=fake_calibrator_builder,
    )

    pooled = arte / "pooled"
    assert (pooled / "scaler.pkl").exists()
    assert (pooled / "pca.pkl").exists()
    assert (pooled / "clusters.pkl").exists()
    assert (pooled / "feature_schema.json").exists()
    assert (pooled / "calibrators" / "RRC.isotonic.pkl").exists()
    assert (pooled / "calibrators" / "BBY.isotonic.pkl").exists()
    rpt = json.loads((pooled / "calibrators" / "calibration_report.json").read_text())
    assert "RRC" in rpt and "BBY" in rpt
