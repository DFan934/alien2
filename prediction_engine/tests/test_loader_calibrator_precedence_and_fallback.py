from pathlib import Path
import json
import joblib
from sklearn.isotonic import IsotonicRegression

from prediction_engine.prediction_engine.artifacts.loader import resolve_artifact_paths, load_calibrator


def _posix(s: str) -> str:
    return str(Path(s)).replace("\\", "/")


def _dump_dummy_iso(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit([0.0, 1.0], [0.0, 1.0])
    joblib.dump(iso, path)


def _seed_core(root: Path):
    (root / "pooled").mkdir(parents=True, exist_ok=True)
    for name in ["scaler.pkl", "pca.pkl", "clusters.pkl", "feature_schema.json", "meta.json"]:
        (root / "pooled" / name).write_text("x")
    (root / "pooled" / "ann").mkdir(parents=True, exist_ok=True)
    for name in ["TREND.index", "RANGE.index", "VOL.index", "GLOBAL.index"]:
        (root / "pooled" / "ann" / name).write_text("x")
    meta_payload = {"payload": {"distance": {"family": "euclidean", "params": {"k_max": 16}}}}
    (root / "pooled" / "meta.json").write_text(json.dumps(meta_payload))


def test_calibrator_precedence_pooled_over_symbol(tmp_path: Path):
    arte = tmp_path / "arte"
    _seed_core(arte)

    # both pooled and per-symbol calibrators exist → choose pooled
    _dump_dummy_iso(arte / "pooled" / "calibrators" / "BBY.isotonic.pkl")
    _dump_dummy_iso(arte / "BBY" / "calibrators" / "BBY.isotonic.pkl")

    paths = resolve_artifact_paths(artifacts_root=arte, symbol="BBY", strategy="pooled")
    assert paths["calibrator_scope"] == "pooled"
    assert _posix(paths["calibrator"]).endswith("pooled/calibrators/BBY.isotonic.pkl")
    assert load_calibrator(paths["calibrator"]) is not None


def test_calibrator_fallback_to_symbol_when_pooled_missing(tmp_path: Path):
    arte = tmp_path / "arte"
    _seed_core(arte)

    # only per-symbol calibrator exists → choose per_symbol
    _dump_dummy_iso(arte / "BBY" / "calibrators" / "BBY.isotonic.pkl")

    paths = resolve_artifact_paths(artifacts_root=arte, symbol="BBY", strategy="pooled")
    assert paths["calibrator_scope"] == "per_symbol"
    assert _posix(paths["calibrator"]).endswith("BBY/calibrators/BBY.isotonic.pkl")
    assert load_calibrator(paths["calibrator"]) is not None


def test_calibrator_missing_reports_scope_missing(tmp_path: Path):
    arte = tmp_path / "arte"
    _seed_core(arte)

    paths = resolve_artifact_paths(artifacts_root=arte, symbol="APA", strategy="pooled")
    assert paths["calibrator_scope"] == "missing"
    assert paths["calibrator"] == ""
    assert load_calibrator(paths["calibrator"]) is None
