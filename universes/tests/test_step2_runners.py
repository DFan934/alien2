import types
import pytest

# We’ll import the modules under test
import importlib

def test_run_backtest_uses_resolver(monkeypatch, tmp_path):
    # Monkeypatch universes.resolve_universe to prove it’s called
    calls = {"count": 0}
    def fake_resolve(cfg, as_of=None):
        calls["count"] += 1
        return ["RRC", "BBY"]
    monkeypatch.setattr("universes.providers.resolve_universe", fake_resolve, raising=True)

    rb = importlib.import_module("scripts.run_backtest")

    # Minimal config: only universe + dates required here
    cfg = {
        "universe": ["rrc", "bby"],
        "parquet_root": tmp_path.as_posix(),  # empty; we just want to reach the resolver & banner
        "start": "1998-08-01",
        "end":   "1998-09-01",
        "universe_max_size": 1000,
    }

    # Run just until it complains about missing parquet data after banner
    with pytest.raises(RuntimeError):
        import asyncio
        asyncio.run(rb.run(cfg))

    assert calls["count"] == 1  # resolver was definitely called

def test_parallel_backtest_uses_resolver(monkeypatch, tmp_path):
    calls = {"count": 0}
    def fake_resolve(cfg, as_of=None):
        calls["count"] += 1
        return ["RRC"]
    monkeypatch.setattr("universes.providers.resolve_universe", fake_resolve, raising=True)

    pb = importlib.import_module("scripts.parallel_backtest")
    cfg = {
        "universe": ["rrc"],
        "parquet_root": tmp_path.as_posix(),
        "start": "1998-08-01",
        "end":   "1998-09-01",
        "gap_pct": 0.02,
        "rvol": 2.0,
        "out_dir": tmp_path.joinpath("out").as_posix(),
        "universe_max_size": 1000,
    }

    # We won't actually execute Dask; just ensure the resolver path is hit before it would run
    with pytest.raises(RuntimeError):
        pb.main(cfg)

    assert calls["count"] == 1



from universes.providers import StaticUniverse, SP500Universe, resolve_universe, UniverseError
import pytest

def test_static_instance_is_accepted():
    cfg = {"universe": StaticUniverse(["RRC", "BBY"])}
    out = resolve_universe(cfg)
    assert out == ["RRC", "BBY"]

def test_sp500_instance_stubs_cleanly():
    cfg = {"universe": SP500Universe(as_of="1999-01-04")}
    with pytest.raises(NotImplementedError):
        resolve_universe(cfg)
