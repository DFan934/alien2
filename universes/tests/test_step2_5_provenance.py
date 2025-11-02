import json
from pathlib import Path
import pytest
import pytest
pytestmark = pytest.mark.phase2_minimal

from prediction_engine.prediction_engine.artifacts.manager import ArtifactManager, _hash_universe  # noqa

def test_universe_hash_changes_meta(tmp_path: Path):
    # Prepare manager roots
    pq = tmp_path / "parquet"
    ar = tmp_path / "artifacts"
    pq.mkdir(); ar.mkdir()

    # Fake hive dirs so _fingerprint_slice sees something (meta rebuild path touched)
    for s in ["RRC", "BBY"]:
        (pq / f"symbol={s}" / "year=1999" / "month=01" / "day=01").mkdir(parents=True, exist_ok=True)
        (pq / f"symbol={s}" / "year=1999" / "month=01" / "day=01" / "part-0.parquet").write_bytes(b"\x50\x41\x52\x31")

    am = ArtifactManager(parquet_root=pq, artifacts_root=ar)

    # 1) Build with one universe
    out1 = am.fit_or_load(universe=["RRC"], start="1999-01-01", end="1999-02-01", strategy="per_symbol", config_hash_parts={})
    meta1 = json.loads((ar / "RRC" / "meta.json").read_text())["payload"]["universe_hash"]

    # 2) Change the universe (add BBY) → expect different hash & rebuild for BBY at least
    out2 = am.fit_or_load(universe=["RRC","BBY"], start="1999-01-01", end="1999-02-01", strategy="per_symbol", config_hash_parts={})
    meta2a = json.loads((ar / "RRC" / "meta.json").read_text())["payload"]["universe_hash"]
    meta2b = json.loads((ar / "BBY" / "meta.json").read_text())["payload"]["universe_hash"]

    assert meta1 != meta2a
    assert meta2a == meta2b == _hash_universe(["RRC","BBY"])

def test_run_header_line(capsys):
    # Minimal imitation of the header; we just verify formatting is correct
    from scripts.run_backtest import _stable_universe_hash, _universe_source_label
    syms = ["RRC","BBY"]
    h = _stable_universe_hash(syms)
    assert len(h) == 12 and all(c in "0123456789abcdef" for c in h)
    src = _universe_source_label({"universe": ["RRC","BBY"]})
    assert src.startswith("static")


def test_run_header_emits_once(monkeypatch, capsys):
    import importlib, asyncio
    rb = importlib.import_module("scripts.run_backtest")

    # Force a tiny config and stub resolve_universe so we don't hit IO
    monkeypatch.setattr("universes.providers.resolve_universe", lambda cfg, as_of=None: ["RRC","BBY"])
    cfg = {"universe": ["RRC","BBY"], "start":"1999-01-01", "end":"1999-02-01", "artifacts_root":"./artifacts"}

    with pytest.raises(RuntimeError):  # expect later code to bail due to empty parquet, etc.
        asyncio.run(rb.run(cfg))

    out = capsys.readouterr().out
    assert out.count("[Run] universe_hash=") == 1
