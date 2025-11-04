# feature_engineering/tests/verify_phase3_helper.py
from pathlib import Path
import json
import pandas as pd
import numpy as np

from prediction_engine.prediction_engine.artifacts.manager import ArtifactManager
from prediction_engine.prediction_engine.artifacts.loader import resolve_artifact_paths, read_distance_contract, load_calibrator

def _seed_parquet(pq: Path, sym: str):
    ts = pd.date_range("1999-01-04 09:30", periods=6, freq="T", tz="UTC").tz_convert(None)
    df = pd.DataFrame({
        "timestamp": ts,
        "symbol": sym,
        "open": np.linspace(10, 10.2, len(ts)),
        "close": np.linspace(10, 10.2, len(ts))
    })
    # Write a tiny dummy parquet hive (only presence matters for manager hooks).
    p = pq / f"symbol={sym}/year=1999/month=01/day=04"
    p.mkdir(parents=True, exist_ok=True)
    (p / "part-0.parquet").write_bytes(b"PAR1")

def test_phase3_end_to_end_pooled_core_two_calibrators(tmp_path):
    # Seed tiny two-symbol hive
    pq = tmp_path / "pq"; pq.mkdir()
    for sym in ["RRC","BBY"]:
        _seed_parquet(pq, sym)

    arte = tmp_path / "arte"
    am = ArtifactManager(parquet_root=pq, artifacts_root=arte)

    # Pooled builder: also drop meta with distance contract
    def pooled(symbols, out_dir, start, end):
        out = Path(out_dir); (out / "ann").mkdir(parents=True, exist_ok=True)
        for name in ["scaler.pkl","pca.pkl","clusters.pkl","feature_schema.json","meta.json"]:
            (out / name).write_text("x")
        (out / "ann" / "GLOBAL.index").write_text("x")
        meta = {"payload":{"distance":{"family":"euclidean","params":{"k_max":8}}}}
        (out / "meta.json").write_text(json.dumps(meta))

    # Calibrator builder: write two .pkl and good metrics
    def cal(sym, pooled_dir, start, end):
        caldir = Path(pooled_dir) / "calibrators"; caldir.mkdir(parents=True, exist_ok=True)
        (caldir / f"{sym}.isotonic.pkl").write_text("x")
        return caldir / f"{sym}.isotonic.pkl", {"ece": 0.02, "brier": 0.21, "pooled_brier": 0.20, "mono_adj_pairs": 9}

    am.fit_or_load(
        universe=["RRC","BBY"],
        start="1999-01-01", end="1999-01-31",
        strategy="pooled",
        config_hash_parts={"metric":"euclidean","k_max":8,"use_isotonic":True},
        schema_hash_parts={"feature_schema_version":"v1"},
        pooled_builder=pooled,
        calibrator_builder=cal,
    )

    # Loader sees the right files for each symbol
    for sym in ["RRC","BBY"]:
        paths = resolve_artifact_paths(artifacts_root=arte, symbol=sym, strategy="pooled")
        assert Path(paths["scaler"]).exists()
        assert Path(paths["clusters"]).exists()
        assert Path(paths["ann_global"]).exists()
        assert paths["calibrator"].endswith(f"pooled/calibrators/{sym}.isotonic.pkl")
        assert load_calibrator(paths["calibrator"]) is not None
        fam, params = read_distance_contract(paths["meta"])
        assert fam == "euclidean" and int(params["k_max"]) == 8
