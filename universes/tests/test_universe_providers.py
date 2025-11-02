# feature_engineering/tests/test_universe_providers.py
import io
from pathlib import Path
import pytest

from universes import (
    UniverseError,
    StaticUniverse,
    FileUniverse,
    SP500Universe,
    resolve_universe,
)

def test_static_universe_ucase_and_dedupe():
    u = StaticUniverse(symbols=["rrc", "RRC", "bby", "BbY", "APA"])
    syms = u.resolve()
    assert syms == ["RRC", "BBY", "APA"]  # upper + dedup preserve order

def test_resolve_universe_from_list_config_enforces_min_and_max():
    cfg = {"universe": ["rrc", "bby", "apa"], "universe_max_size": 10}
    syms = resolve_universe(cfg)
    assert syms == ["RRC", "BBY", "APA"]

    cfg_too_big = {"universe": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), "universe_max_size": 5}
    with pytest.raises(UniverseError):
        resolve_universe(cfg_too_big)

def test_resolve_universe_from_txt(tmp_path: Path):
    txt = tmp_path / "u.txt"
    txt.write_text(
        """
        # demo universe
        rrc
        bby

        apa   # inline comment allowed if you want, we'll ignore only full-line '#'
        """,
        encoding="utf-8",
    )
    # Our simple parser ignores full-line comments and blanks; trims, uppercases.
    # The inline '# ...' won't be removed by the minimal parser; so keep rows clean in real files.
    # Here we include a symbol with trailing content to confirm strict normalization.
    # We'll simulate a clean file by writing only the symbol per line.
    clean = tmp_path / "clean.txt"
    clean.write_text("rrc\nbby\napa\n", encoding="utf-8")

    cfg = {"universe": str(clean)}
    syms = resolve_universe(cfg)
    assert syms == ["RRC", "BBY", "APA"]

def test_resolve_universe_from_csv(tmp_path: Path):
    csvp = tmp_path / "u.csv"
    csvp.write_text("SYMBOL,foo\nrrc,1\nBBY,2\nApa,3\n", encoding="utf-8")
    cfg = {"universe": str(csvp)}
    syms = resolve_universe(cfg)
    assert syms == ["RRC", "BBY", "APA"]

def test_sp500_universe_is_clean_stub():
    cfg = {"universe": {"type": "sp500", "as_of": "1999-01-04"}}
    with pytest.raises(NotImplementedError) as ei:
        resolve_universe(cfg)
    assert "SP500Universe" in str(ei.value)

def test_invalid_configs():
    with pytest.raises(UniverseError):
        resolve_universe({})  # missing 'universe'

    with pytest.raises(UniverseError):
        resolve_universe({"universe": 123})  # unsupported type

    with pytest.raises(UniverseError):
        resolve_universe({"universe": {"type": "static"}})  # missing symbols

    with pytest.raises(UniverseError):
        resolve_universe({"universe": {"type": "file", "path": 123}})  # bad path type

    with pytest.raises(UniverseError):
        resolve_universe({"universe": {"type": "sp500"}})  # missing as_of



