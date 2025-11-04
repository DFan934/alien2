from pathlib import Path
from prediction_engine.prediction_engine.artifacts.manager import ArtifactManager

def test_pooled_ann_layout(tmp_path, monkeypatch):
    pq = tmp_path / "parquet"; pq.mkdir()
    arte = tmp_path / "artifacts"

    # Fake pooled builder that writes the required core files
    def fake_pooled(symbols, out_dir, start, end):
        out = Path(out_dir)
        for name in ["scaler.pkl", "pca.pkl", "clusters.pkl", "feature_schema.json"]:
            (out / name).write_text("x", encoding="utf-8")

    am = ArtifactManager(parquet_root=pq, artifacts_root=arte)
    am.fit_or_load(
        universe=["RRC","BBY"],
        start="1999-01-01", end="1999-02-01",
        strategy="pooled",
        config_hash_parts={"metric": "euclidean"},
        schema_hash_parts={"feature_schema_version": "v1"},
        pooled_builder=fake_pooled,
    )

    ann = arte / "pooled" / "ann"
    assert (ann / "TREND.index").exists()
    assert (ann / "RANGE.index").exists()
    assert (ann / "VOL.index").exists()
    assert (ann / "GLOBAL.index").exists()
