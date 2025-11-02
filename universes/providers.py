# universes/providers.py
"""
Universe Providers
------------------
Centralizes how we determine the set of symbols ("universe") to backtest.

Usage patterns (examples):
- Static list:
    cfg = {"universe": ["RRC", "bby"], "universe_max_size": 1000}
    symbols = resolve_universe(cfg)

- From file (TXT or CSV):
    # TXT: one symbol per line (comments allowed with '#' and blank lines ignored)
    # CSV: requires a 'symbol' column (case-insensitive)
    cfg = {"universe": "path/to/universe.txt"}
    symbols = resolve_universe(cfg)

- SP500 point-in-time (stubbed for now):
    cfg = {"universe": {"type": "sp500", "as_of": "1999-01-04"}}
    symbols = resolve_universe(cfg)   # raises NotImplementedError (clean message)

Design notes:
- Returns a **de-duplicated, UPPER-cased** list of strings.
- Enforces a minimum length (>=1) and an optional maximum size guardrail.
- Does no data-lake / parquet I/O (dry and fast). Only reads local file if FileUniverse is used.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Union, Dict

__all__ = [
    "UniverseError",
    "StaticUniverse",
    "FileUniverse",
    "SP500Universe",
    "resolve_universe",
]

# ---------------------------
# Exceptions
# ---------------------------

class UniverseError(ValueError):
    """Raised for invalid universe inputs or guardrail violations."""
    pass

# ---------------------------
# Helpers
# ---------------------------

def _normalize_symbol(s: str) -> str:
    if not isinstance(s, str):
        raise UniverseError(f"Symbol must be a string, got {type(s)}")
    s = s.strip()
    if not s:
        raise UniverseError("Empty symbol encountered after stripping whitespace")
    return s.upper()

def _dedupe_preserve_order(symbols: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def _read_txt_symbols(path: Path) -> List[str]:
    if not path.exists():
        raise UniverseError(f"Universe file does not exist: {path}")
    out: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(_normalize_symbol(line))
    if not out:
        raise UniverseError(f"No symbols found in TXT file: {path}")
    return _dedupe_preserve_order(out)

def _read_csv_symbols(path: Path) -> List[str]:
    if not path.exists():
        raise UniverseError(f"Universe file does not exist: {path}")
    # Minimal CSV reader to avoid pandas dependency here.
    # Assumes comma-separated; trims whitespace; case-insensitive 'symbol' header.
    import csv
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise UniverseError(f"CSV has no header row: {path}")
        # Find the 'symbol' column case-insensitively
        cols = {c.lower(): c for c in reader.fieldnames}
        if "symbol" not in cols:
            raise UniverseError(f"CSV is missing required 'symbol' column: {path}")
        sym_col = cols["symbol"]
        out: List[str] = []
        for row in reader:
            raw = row.get(sym_col, "")
            raw = "" if raw is None else str(raw)
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            out.append(_normalize_symbol(raw))
    if not out:
        raise UniverseError(f"No symbols found in CSV file: {path}")
    return _dedupe_preserve_order(out)

# ---------------------------
# Providers
# ---------------------------

@dataclass(frozen=True)
class StaticUniverse:
    symbols: List[str]

    def resolve(self) -> List[str]:
        normalized = [_normalize_symbol(s) for s in self.symbols]
        return _dedupe_preserve_order(normalized)

@dataclass(frozen=True)
class FileUniverse:
    path: Union[str, Path]

    def resolve(self) -> List[str]:
        p = Path(self.path)
        ext = p.suffix.lower()
        if ext == ".txt":
            return _read_txt_symbols(p)
        elif ext == ".csv":
            return _read_csv_symbols(p)
        else:
            raise UniverseError(f"Unsupported file extension '{ext}' for universe file: {p}")

@dataclass(frozen=True)
class SP500Universe:
    as_of: str  # ISO-like date string, e.g., "1999-01-04"

    def resolve(self) -> List[str]:
        # Placeholder: in Phase 2 we only provide a clean, explicit stub.
        # This keeps the interface stable so Phase 2.5 can swap in PIT logic.
        raise NotImplementedError(
            "SP500Universe(as_of=...) is not implemented yet. "
            "Provide a StaticUniverse or FileUniverse for now."
        )

# ---------------------------
# Resolver entry point
# ---------------------------

def resolve_universe(config: Dict, *, as_of: Optional[str] = None) -> List[str]:
    """
    Resolve a universe from CONFIG and optional as_of date.

    Accepted CONFIG["universe"] forms:
      1) List[str]                      -> StaticUniverse(symbols=...)
      2) str path to .txt/.csv          -> FileUniverse(path=...)
      3) dict:
           {"type": "static", "symbols": [...]}
           {"type": "file", "path": "path/to/file.csv"}
           {"type": "sp500", "as_of": "YYYY-MM-DD"}  # stubbed for Phase 2

    Guardrails:
      - Min length >= 1
      - Optional max size via CONFIG["universe_max_size"] (int)
    """
    if "universe" not in config:
        raise UniverseError("CONFIG missing required key: 'universe'")

    udef = config["universe"]



    # Build a provider instance from the definition
    #provider: Union[StaticUniverse, FileUniverse, SP500Universe]



    '''if isinstance(udef, list):
        provider = StaticUniverse(symbols=udef)
    elif isinstance(udef, str):
        provider = FileUniverse(path=udef)
    elif isinstance(udef, dict):'''
    # Accept provider instances directly (e.g., StaticUniverse([...]))
    if isinstance(udef, (StaticUniverse, FileUniverse, SP500Universe)):
        provider = udef
    # Otherwise, build a provider from common literals
    elif isinstance(udef, list):
        provider = StaticUniverse(symbols=udef)
    elif isinstance(udef, str):
        provider = FileUniverse(path=udef)
    elif isinstance(udef, dict):
        utype = str(udef.get("type", "")).lower()
        if utype == "static":
            syms = udef.get("symbols")
            if not isinstance(syms, list):
                raise UniverseError("For {'type':'static'}, provide a list under 'symbols'")
            provider = StaticUniverse(symbols=syms)
        elif utype == "file":
            path = udef.get("path")
            if not isinstance(path, str):
                raise UniverseError("For {'type':'file'}, provide a string path under 'path'")
            provider = FileUniverse(path=path)
        elif utype == "sp500":
            # Prefer passed `as_of` param if present; else try dict
            date = as_of or udef.get("as_of")
            if not isinstance(date, str) or not date:
                raise UniverseError("For {'type':'sp500'}, provide an 'as_of' date string")
            provider = SP500Universe(as_of=date)
        else:
            raise UniverseError(f"Unknown universe type in dict: {utype!r}")
    else:
        raise UniverseError(f"Unsupported universe definition type: {type(udef)}")

    # Resolve and normalize
    try:
        symbols = provider.resolve()
    except NotImplementedError as e:
        # Surface a clean message (does not crash the process unexpectedly)
        # Re-raise so callers/tests can assert on it explicitly.
        raise e

    if len(symbols) < 1:
        raise UniverseError("Resolved universe is empty")

    # Enforce max size if provided
    max_size = config.get("universe_max_size")
    if isinstance(max_size, int) and max_size > 0:
        if len(symbols) > max_size:
            raise UniverseError(
                f"Resolved universe size {len(symbols)} exceeds universe_max_size={max_size}"
            )

    return symbols
