from pathlib import Path
import json

from prediction_engine.prediction_engine.artifacts.loader import resolve_artifact_paths, read_distance_contract


def _posix(s: str) -> str:
    return str(Path(s)).replace("\\", "/")


def test_per_symbol_strategy_paths_and_contract(tmp_path: Path):
    root = tmp_path / "arte"
    # per-symbol core layout
    sym = "RRC"
    (root / sym / "ann").mkdir(parents=True, exist_ok=True)
    for name in ["scaler.pkl", "pca.pkl", "clusters.pkl", "feature_schema.json", "meta.json"]:
        (root / sym / name).write_text("x")
    for name in ["TREND.index", "RANGE.index", "VOL.index", "GLOBAL.index"]:
        (root / sym / "ann" / name).write_text("x")
    # per-symbol calibrator only
    (root / sym / "calibrators").mkdir(parents=True, exist_ok=True)
    (root / sym / "calibrators" / f"{sym}.isotonic.pkl").write_text("x")
    # distance contract
    (root / sym / "meta.json").write_text(
        json.dumps({"payload": {"distance": {"family": "rf_weighted", "params": {"k_max": 8, "rf_weights_len": 5}}}})
    )

    paths = resolve_artifact_paths(artifacts_root=root, symbol=sym, strategy="per_symbol")
    # core_dir is a directory; ensure it resolves to the symbol leaf
    assert Path(paths["core_dir"]).name == sym
    assert _posix(paths["scaler"]).endswith(f"{sym}/scaler.pkl")
    assert _posix(paths["ann_trend"]).endswith(f"{sym}/ann/TREND.index")
    assert paths["calibrator_scope"] == "per_symbol"
    assert _posix(paths["calibrator"]).endswith(f"{sym}/calibrators/{sym}.isotonic.pkl")

    fam, params = read_distance_contract(paths["meta"])
    assert fam == "rf_weighted" and int(params["k_max"]) == 8 and int(params["rf_weights_len"]) == 5
