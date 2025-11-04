from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
import joblib

from prediction_engine.testing_validation.walkforward import WalkForwardRunner


def _write_parquet_day(base: Path, sym: str, y: int, m: int, d: int,
                       start: str, periods: int = 30) -> None:
    """Write a valid parquet under hive partitions for one day."""
    p = base / f"symbol={sym}" / f"year={y}" / f"month={m:02d}" / f"day={d:02d}"
    p.mkdir(parents=True, exist_ok=True)
    ts = pd.date_range(start, periods=periods, freq="T", tz="UTC").tz_convert(None)
    df = pd.DataFrame({
        "timestamp": ts,
        "symbol": sym,
        "open":   np.linspace(10, 10 + 0.01*(periods-1), periods),
        "high":   np.linspace(10.01, 10.01 + 0.01*(periods-1), periods),
        "low":    np.linspace(9.99, 9.99 + 0.01*(periods-1), periods),
        "close":  np.linspace(10, 10 + 0.01*(periods-1), periods),
        "volume": 1000,
    })
    df.to_parquet(p / "part-0.parquet", engine="pyarrow", index=False)


def _mk_cross_month_df(sym: str, base: float,
                       train_minutes: int = 200, test_minutes: int = 60) -> pd.DataFrame:
    """
    Build an in-memory slice with a *long* train segment ending <= 1999-01-31 23:59
    and a short test segment on 1999-02-01. This guarantees train samples >= 64.
    """
    # Train: 200 minutes ending right before midnight Jan 31
    ts_train = pd.date_range("1999-01-31 20:10", periods=train_minutes, freq="T", tz="UTC").tz_convert(None)
    # Test: 60 minutes starting Feb 1 00:00
    ts_test  = pd.date_range("1999-02-01 00:00", periods=test_minutes,  freq="T", tz="UTC").tz_convert(None)
    ts = ts_train.append(ts_test)
    n = ts.size
    return pd.DataFrame({
        "timestamp": ts,
        "symbol": sym,
        "open":  base + np.linspace(0, 0.001*(n-1), n),
        "high":  base + 0.01 + np.linspace(0, 0.001*(n-1), n),
        "low":   base - 0.01 + np.linspace(0, 0.001*(n-1), n),
        "close": base + np.linspace(0, 0.001*(n-1), n),
        "volume": 1000,
    })


def test_walkforward_two_symbols_smoke(tmp_path: Path):
    """
    Acceptance: Smoke test with 2 symbols. Verifies the runner can score both symbols and
    that manifests reference the calibrators and distance contract.
    """
    # --- Arrange EV artifacts (pooled) ---
    ev_root = tmp_path / "ev_artefacts" / "pooled"
    (ev_root / "ann").mkdir(parents=True, exist_ok=True)
    (ev_root / "calibrators").mkdir(parents=True, exist_ok=True)

    (ev_root / "feature_schema.json").write_text(json.dumps({"feature_cols": []}))
    meta = {"payload": {"distance": {"family": "euclidean", "params": {"k_max": 16}}}}
    (ev_root / "meta.json").write_text(json.dumps(meta))
    for name in ["TREND.index", "RANGE.index", "VOL.index", "GLOBAL.index"]:
        (ev_root / "ann" / name).write_text("x")

    for sym in ["RRC", "BBY"]:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit([0.0, 1.0], [0.0, 1.0])
        joblib.dump(iso, ev_root / "calibrators" / f"{sym}.isotonic.pkl")

    # --- Seed a real parquet hive (two days are enough for the script's checks) ---
    pq = tmp_path / "pq"
    for sym in ["RRC", "BBY"]:
        _write_parquet_day(pq, sym, 1999, 1, 31, "1999-01-31 23:30", periods=30)
        _write_parquet_day(pq, sym, 1999, 2,  1, "1999-02-01 00:00", periods=30)

    # --- In-memory data with a sufficiently long train segment (>=64 samples) ---
    df_rrc = _mk_cross_month_df("RRC", 10.0, train_minutes=200, test_minutes=60)
    df_bby = _mk_cross_month_df("BBY", 20.0, train_minutes=200, test_minutes=60)

    start = pd.Timestamp("1999-01-01 00:00")
    end   = pd.Timestamp("1999-02-01 00:59")

    for sym, df in [("RRC", df_rrc), ("BBY", df_bby)]:
        runner = WalkForwardRunner(
            artifacts_root=tmp_path / f"wf_{sym}",
            parquet_root=pq,
            ev_artifacts_root=ev_root.parent,  # contains "pooled"
            symbol=sym,
            horizon_bars=1,
            longest_lookback_bars=1,
            debug_no_costs=True,
        )
        out = runner.run(df_full=df, df_scanned=df, start=start, end=end)
        assert out["total_test_rows"] >= 1

        m = json.loads((tmp_path / f"wf_{sym}" / "fold_01" / "manifest.json").read_text())
        cal = m["artifact_sources"]["calibrator"]["path"].replace("\\", "/")
        assert cal.endswith(f"pooled/calibrators/{sym}.isotonic.pkl")
        assert m["artifact_sources"]["distance_contract"]["family"] == "euclidean"
