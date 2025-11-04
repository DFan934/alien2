# prediction_engine/tests/test_runbacktest_schema_passthrough.py
import asyncio
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import run_backtest as rb
from prediction_engine.prediction_engine.artifacts.manager import ArtifactManager


def _dummy_bars(symbol: str, n=7):
    ts = pd.date_range("2019-05-01 09:30", periods=n, freq="T", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open": np.linspace(10, 10.4, n),
        "high": np.linspace(10.1, 10.5, n),
        "low":  np.linspace( 9.9, 10.3, n),
        "close":np.linspace(10, 10.4, n),
        "volume": np.full(n, 1000, dtype=int),
        "symbol": symbol,
    })


def test_runbacktest_passes_schema_hash_parts(monkeypatch, tmp_path):
    captured = {}

    # --- Spy ArtifactManager.fit_or_load ---
    def spy_fit_or_load(self, **kwargs):
        captured["kwargs"] = kwargs
        return None
    monkeypatch.setattr(ArtifactManager, "fit_or_load", spy_fit_or_load, raising=True)

    # --- Make the module lightweight regardless of which entrypoint it uses ---
    # 1) Bars loader
    monkeypatch.setattr(rb, "_load_bars_for_symbol", lambda cfg, symbol: _dummy_bars(symbol), raising=False)

    # 2) Resolve path → keep everything under tmp_path and create on demand
    def fake_resolve(path_like, *, create=False, is_dir=None):
        p = Path(path_like)
        if not p.is_absolute():
            p = tmp_path / p
        if create:
            if is_dir or (is_dir is None and (str(p).endswith("/") or str(p).endswith("\\"))):
                p.mkdir(parents=True, exist_ok=True)
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
        return p
    monkeypatch.setattr(rb, "_resolve_path", fake_resolve, raising=False)

    # 3) Scanner/detector → accept all rows
    class _Detector:
        mode = "OR"
        async def __call__(self, df):  # async to match real interface
            return np.ones(len(df), dtype=bool)
    monkeypatch.setattr(rb, "build_detectors", lambda **_: _Detector(), raising=False)

    # 4) WalkForwardRunner → no-op
    class _WFR:
        def __init__(self, *a, **k): pass
        def run(self, **k): return {}
    monkeypatch.setattr(rb, "WalkForwardRunner", _WFR, raising=False)

    # 5) Report (if present) → no-op
    if hasattr(rb, "generate_report"):
        monkeypatch.setattr(rb, "generate_report", lambda *a, **k: None, raising=False)

    # --- Minimal config containing the schema bits we assert ---
    cfg = {
        "start": "1999-05-01",
        "end":   "1999-05-31",
        "universe": ["RRC"],
        "parquet_root": str(tmp_path / "parquet"),
        "artifacts_root": str(tmp_path / "artifacts"),
        "artefacts": str(tmp_path / "weights"),
        "calibration_dir": str(tmp_path / "weights" / "calibration"),
        "horizon_bars": 20,
        "feature_schema_version": "vX",
        "feature_schema_list": ["vwap_z", "rvol_20", "ema9_slope"],
        "metric": "mahalanobis",
        "regime_settings": {"modes": ["TREND", "RANGE", "VOL"]},
        "p_gate_quantile": 0.55,
        "full_p_quantile": 0.65,
        "sign_check": False,
    }

    # --- Find an entrypoint and call it correctly (positional vs keyword, sync vs async) ---
    entry = getattr(rb, "_run_one_symbol", None) or getattr(rb, "_run_one_symbol_async", None)
    assert entry is not None, "No _run_one_symbol(_async) entrypoint found"

    is_async = asyncio.iscoroutinefunction(entry)
    params = list(inspect.signature(entry).parameters.keys())

    if params and params[0] in {"sym", "symbol"}:
        # style: (sym, cfg, ...)
        if is_async:
            asyncio.run(entry("RRC", cfg))
        else:
            entry("RRC", cfg)
    else:
        # style: (cfg, sym, ...)
        if is_async:
            asyncio.run(entry(cfg, "RRC",
                              arte_root=cfg["artifacts_root"],
                              parquet_root=cfg["parquet_root"]))
        else:
            entry(cfg, "RRC",
                  arte_root=cfg["artifacts_root"],
                  parquet_root=cfg["parquet_root"])

    # --- Assertions on the captured kwargs ---
    kw = captured.get("kwargs", {})
    assert "config_hash_parts" in kw and "schema_hash_parts" in kw, "fit_or_load must receive both hash dicts"
    shp = kw["schema_hash_parts"]
    assert shp.get("feature_schema_version") == "vX"
    assert shp.get("label_horizon_bars") == 20
    assert shp.get("regime_settings", {}).get("modes") == ["TREND", "RANGE", "VOL"]
