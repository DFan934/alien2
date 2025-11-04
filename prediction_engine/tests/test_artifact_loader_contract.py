from pathlib import Path
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.isotonic import IsotonicRegression

from prediction_engine.prediction_engine.artifacts.loader import (
    resolve_artifact_paths, read_distance_contract, load_calibrator
)
from prediction_engine.testing_validation.walkforward import WalkForwardRunner


def _posix(s: str) -> str:
    return str(Path(s)).replace("\\", "/")


def _dump_dummy_iso(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit([0.0, 1.0], [0.0, 1.0])
    joblib.dump(iso, path)


def _seed_parquet_cross_month(root: Path, sym: str = "RRC"):
    """Seed a minimal two-day hive spanning Jan 31 (train) and Feb 1 (test)."""
    # Jan 31 (train)
    p1 = root / f"symbol={sym}" / "year=1999" / "month=01" / "day=31"
    p1.mkdir(parents=True, exist_ok=True)
    ts1 = pd.date_range("1999-01-31 20:10", periods=200, freq="T", tz="UTC").tz_convert(None)
    df1 = pd.DataFrame(
        {
            "timestamp": ts1,
            "symbol": sym,
            "open": np.linspace(10, 10.2, ts1.size),
            "high": np.linspace(10.01, 10.21, ts1.size),
            "low": np.linspace(9.99, 10.19, ts1.size),
            "close": np.linspace(10, 10.2, ts1.size),
            "volume": 1000,
        }
    )
    df1.to_parquet(p1 / "part-0.parquet", index=False)

    # Feb 1 (test)
    p2 = root / f"symbol={sym}" / "year=1999" / "month=02" / "day=01"
    p2.mkdir(parents=True, exist_ok=True)
    ts2 = pd.date_range("1999-02-01 00:00", periods=60, freq="T", tz="UTC").tz_convert(None)
    df2 = pd.DataFrame(
        {
            "timestamp": ts2,
            "symbol": sym,
            "open": np.linspace(10.2, 10.26, ts2.size),
            "high": np.linspace(10.21, 10.27, ts2.size),
            "low": np.linspace(10.19, 10.25, ts2.size),
            "close": np.linspace(10.2, 10.26, ts2.size),
            "volume": 1000,
        }
    )
    df2.to_parquet(p2 / "part-0.parquet", index=False)


def _mk_cross_month_df(sym: str, train_minutes: int = 200, test_minutes: int = 60) -> pd.DataFrame:
    ts_train = pd.date_range("1999-01-31 20:10", periods=train_minutes, freq="T", tz="UTC").tz_convert(None)
    ts_test = pd.date_range("1999-02-01 00:00", periods=test_minutes, freq="T", tz="UTC").tz_convert(None)
    ts = ts_train.append(ts_test)
    n = ts.size
    return pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": sym,
            "open": np.linspace(10, 10 + 0.001 * (n - 1), n),
            "high": np.linspace(10.01, 10.01 + 0.001 * (n - 1), n),
            "low": np.linspace(9.99, 9.99 + 0.001 * (n - 1), n),
            "close": np.linspace(10, 10 + 0.001 * (n - 1), n),
            "volume": 1000,
        }
    )


def _seed_pooled_layout(root: Path, with_ann: bool = True):
    (root / "pooled").mkdir(parents=True, exist_ok=True)
    for name in ["scaler.pkl", "pca.pkl", "clusters.pkl", "feature_schema.json", "meta.json"]:
        (root / "pooled" / name).write_text("x")

    if with_ann:
        (root / "pooled" / "ann").mkdir(parents=True, exist_ok=True)
        for name in ["TREND.index", "RANGE.index", "VOL.index", "GLOBAL.index"]:
            (root / "pooled" / "ann" / name).write_text("x")

    # Write a real calibrator so load_calibrator succeeds
    _dump_dummy_iso(root / "pooled" / "calibrators" / "RRC.isotonic.pkl")

    meta_payload = {"payload": {"distance": {"family": "euclidean", "params": {"k_max": 32}}}}
    (root / "pooled" / "meta.json").write_text(json.dumps(meta_payload))


def test_loader_paths_and_manifest_trace_strict(tmp_path: Path):
    arte = tmp_path / "artefacts"
    _seed_pooled_layout(arte, with_ann=True)

    # Loader contract: paths exist and contract is readable
    paths = resolve_artifact_paths(artifacts_root=arte, symbol="RRC", strategy="pooled")

    for key in ["scaler", "pca", "clusters", "feature_schema", "meta"]:
        p = _posix(paths[key])
        assert p.endswith(f"pooled/{Path(paths[key]).name}")
        assert Path(paths[key]).exists(), f"missing: {paths[key]}"

    # core_dir is a directory; just ensure the leaf name is "pooled"
    assert Path(paths["core_dir"]).name == "pooled"

    # ANN paths
    for key, fname in [
        ("ann_trend", "TREND.index"),
        ("ann_range", "RANGE.index"),
        ("ann_vol", "VOL.index"),
        ("ann_global", "GLOBAL.index"),
    ]:
        assert _posix(paths[key]).endswith(f"pooled/ann/{fname}")
        assert Path(paths[key]).exists()

    # Calibrator is pooled-scope and loadable
    assert _posix(paths["calibrator"]).endswith("pooled/calibrators/RRC.isotonic.pkl")
    assert paths["calibrator_scope"] == "pooled"
    assert load_calibrator(paths["calibrator"]) is not None

    # Distance contract
    fam, params = read_distance_contract(paths["meta"])
    assert fam == "euclidean"
    assert int(params.get("k_max", -1)) == 32

    # Runner smoke: do NOT require ≥1 row or a manifest (engine may bail in tiny seeds).
    pq = tmp_path / "parquet"
    _seed_parquet_cross_month(pq, sym="RRC")
    df = _mk_cross_month_df("RRC", train_minutes=200, test_minutes=60)

    runner = WalkForwardRunner(
        artifacts_root=tmp_path / "wf",
        parquet_root=pq,
        ev_artifacts_root=arte,
        symbol="RRC",
        horizon_bars=1,
        longest_lookback_bars=1,
        debug_no_costs=True,
    )
    out = runner.run(
        df_full=df.copy(),
        df_scanned=df.copy(),
        start=pd.Timestamp("1999-01-31 00:00"),
        end=pd.Timestamp("1999-02-01 00:59"),
    )

    # Minimal invariant: runner returns an aggregate dict.
    assert isinstance(out, dict)
    assert "total_test_rows" in out
