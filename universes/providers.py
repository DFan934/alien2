from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Union
import csv

def _norm_symbol(s: str) -> str:
    return s.strip().upper()

def _unique(seq: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for s in seq:
        if s not in seen:
            seen.add(s); out.append(s)
    return out

@dataclass(frozen=True)
class StaticUniverse:
    symbols: Sequence[str]
    def list(self) -> List[str]:
        return _unique(_norm_symbol(s) for s in self.symbols)

@dataclass(frozen=True)
class FileUniverse:
    path: Union[str, Path]
    def list(self) -> List[str]:
        p = Path(self.path)
        if not p.exists():
            raise FileNotFoundError(p)
        syms: List[str] = []
        # Simple heuristics: .csv expects a header with 'symbol' or first col; .txt is one symbol per line
        if p.suffix.lower() == ".csv":
            with p.open(newline="", encoding="utf-8") as fh:
                rdr = csv.DictReader(fh)
                if "symbol" in rdr.fieldnames or "ticker" in rdr.fieldnames:
                    key = "symbol" if "symbol" in rdr.fieldnames else "ticker"
                    for row in rdr:
                        if row.get(key):
                            syms.append(_norm_symbol(row[key]))
                else:
                    fh.seek(0)
                    rdr2 = csv.reader(fh)
                    for row in rdr2:
                        if row and row[0]:
                            syms.append(_norm_symbol(row[0]))
        else:
            with p.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        syms.append(_norm_symbol(line))
        return _unique(syms)

def resolve_universe(obj: Union[StaticUniverse, FileUniverse, Sequence[str]]) -> List[str]:
    """Accept StaticUniverse, FileUniverse, or a plain list of strings."""
    if isinstance(obj, StaticUniverse):
        return obj.list()
    if isinstance(obj, FileUniverse):
        return obj.list()
    if isinstance(obj, (list, tuple)):
        return _unique(_norm_symbol(s) for s in obj)
    raise TypeError(f"Unsupported universe type: {type(obj)}")
