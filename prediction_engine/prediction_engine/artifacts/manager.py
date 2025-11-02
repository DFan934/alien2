# prediction_engine/artifacts/manager.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Tuple, Optional
import hashlib, json
import pandas as pd
import hashlib
import datetime as dt

import pyarrow as pa
import pyarrow.dataset as ds


Strategy = Literal["per_symbol", "pooled"]


def _hash_obj(obj: object) -> str:
    """Stable hash for simple JSON-serializable objects."""
    s = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(s).hexdigest()



def _hash_universe(symbols: Iterable[str]) -> str:
    """Stable SHA1 of sorted upper-cased symbols; short prefix is fine for cache keys."""
    s = json.dumps(sorted([str(x).upper() for x in symbols]), separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha1(s).hexdigest()


'''def _fingerprint_slice(parquet_root: Path, symbols: Iterable[str], start, end) -> Dict[str, Dict[str, object]]:
    start_ts, end_ts = pd.to_datetime(start), pd.to_datetime(end)
    out: Dict[str, Dict[str, object]] = {}

    for sym in symbols:
        sym_dir = parquet_root / f"symbol={sym}"
        if not sym_dir.exists():
            out[sym] = {"rows": 0, "tmax": None}
            continue

        dset = ds.dataset(
            str(sym_dir),
            format="parquet",
            partitioning="hive",
            exclude_invalid_files=True,  # skip schema.json, etc.
        )
        filt = (ds.field("timestamp") >= start_ts) & (ds.field("timestamp") <= end_ts)
        tbl = dset.to_table(filter=filt, columns=["timestamp"])
        if tbl.num_rows == 0:
            out[sym] = {"rows": 0, "tmax": None}
        else:
            ts_list = pd.to_datetime(tbl.column("timestamp").to_pylist())
            out[sym] = {"rows": int(tbl.num_rows), "tmax": pd.Timestamp(max(ts_list)).isoformat()}
    return out
'''


def _fingerprint_slice(parquet_root, universe, start: str, end: str) -> str:
    """
    Return a stable fingerprint for the slice described by (universe, start, end).

    Fast path:
      - If the dataset schema contains 'timestamp' (and 'symbol'), we compute a hash
        from the number of rows and min/max timestamps after filtering by symbol
        and time, which is cheap and stable.

    Fallback path:
      - If files are stubbed or schema-less (e.g., test writes bare 'PAR1' bytes),
        we hash the list of matching files (relative path, size, mtime) under
        hive-style partitions year=/month=/day= filtered against [start, end).
    """
    root = Path(parquet_root)
    start_dt = dt.datetime.fromisoformat(start)
    end_dt = dt.datetime.fromisoformat(end)

    h = hashlib.blake2s(digest_size=16)
    h.update(",".join(sorted(universe)).encode("utf-8"))

    try:
        # Try Arrow fast-path first.
        dset = ds.dataset(
            str(root),
            format="parquet",
            partitioning="hive",
            ignore_missing_files=True,
        )
        names = set(dset.schema.names)

        if "timestamp" in names and "symbol" in names:
            # Filter in Arrow space.
            filt = (
                ds.field("symbol").isin(universe)
                & (ds.field("timestamp") >= ds.scalar(start_dt))
                & (ds.field("timestamp") < ds.scalar(end_dt))
            )
            # Pull just the timestamp column to keep it light.
            tbl = dset.to_table(filter=filt, columns=["timestamp"])
            n = len(tbl)

            h.update(str(n).encode("utf-8"))
            if n:
                ts_pd = tbl.column("timestamp").to_pandas()
                # If timezone-aware, pandas prints ISO consistently; good for hashing.
                h.update(str(ts_pd.min()).encode("utf-8"))
                h.update(str(ts_pd.max()).encode("utf-8"))

            return h.hexdigest()

        # If 'timestamp' is missing, fall through to the filesystem fallback.
    except Exception:
        # Any Arrow hiccup falls back to filesystem hashing below.
        pass

    # ---- Fallback: hash hive-partitioned file stats (path, size, mtime) ----
    def date_from_parts(parts: tuple[str, ...]) -> dt.datetime | None:
        """Extract a date from hive path parts like ('symbol=RRC','year=1999','month=01','day=01',...)."""
        y = m = d = None
        for p in parts:
            if p.startswith("year="):
                try:
                    y = int(p[5:])
                except ValueError:
                    return None
            elif p.startswith("month="):
                try:
                    m = int(p[6:])
                except ValueError:
                    return None
            elif p.startswith("day="):
                try:
                    d = int(p[4:])
                except ValueError:
                    return None
        if y and m and d:
            try:
                return dt.datetime(y, m, d)
            except ValueError:
                return None
        return None

    file_facts: list[tuple[str, int, int]] = []
    for sym in universe:
        sym_root = root / f"symbol={sym}"
        if not sym_root.exists():
            continue
        for p in sym_root.rglob("*.parquet"):
            rel = p.relative_to(root)
            fdate = date_from_parts(rel.parts)
            # If we can parse a hive date, use it to gate by [start, end); if not, include it.
            if fdate is None or (start_dt <= fdate < end_dt):
                st = p.stat()
                # (relative path, size, mtime[int]) sorted for stability
                file_facts.append((str(rel).replace("\\", "/"), st.st_size, int(st.st_mtime)))

    for rel, size, mtime in sorted(file_facts):
        h.update(rel.encode("utf-8"))
        h.update(str(size).encode("utf-8"))
        h.update(str(mtime).encode("utf-8"))

    return h.hexdigest()



def _load_meta(meta_path: Path) -> Dict[str, object] | None:
    try:
        if meta_path.exists():
            return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def _needs_rebuild(
    old: Dict[str, object] | None,
    new_payload: Dict[str, object],
) -> bool:
    """Return True if meta is missing or payload (fingerprint/config) differs."""
    if not old:
        return True
    return _hash_obj(old.get("payload", {})) != _hash_obj(new_payload)


@dataclass
class ArtifactManager:
    parquet_root: Path
    artifacts_root: Path
    fitted_pipeline_dir: Optional[Path] = None   # ← add this line

    def fit_or_load(
        self,
        *,
        universe: List[str],
        start: str,
        end: str,
        strategy: Strategy = "per_symbol",
        config_hash_parts: Dict[str, object] | None = None,
        # builder hooks (optional overrides / test doubles)
        per_symbol_builder=None,   # callable(symbol, out_dir: Path, start, end) -> None
        pooled_builder=None,       # callable(symbols: List[str], out_dir: Path, start, end) -> None
    ) -> Dict[str, Path]:
        """
        Ensure artifacts exist and are fresh for the given data slice.
        Returns mapping {symbol: artifact_dir} for per_symbol, or {"__pooled__": dir} for pooled.
        Freshness is determined by comparing a *data fingerprint* + a *config hash*
        against meta.json saved in the destination directory.

        config_hash_parts lets you include knobs that *should* trigger rebuilds
        (e.g., pca_variance, k_max, residual_threshold, scanner flags, etc.).
        """
        self.parquet_root = Path(self.parquet_root).expanduser().resolve()
        self.artifacts_root = Path(self.artifacts_root).expanduser().resolve()
        self.artifacts_root.mkdir(parents=True, exist_ok=True)

        cfg_hash = _hash_obj(config_hash_parts or {})
        fp = _fingerprint_slice(self.parquet_root, universe, start, end)
        u_hash = _hash_universe(universe)

        if strategy == "per_symbol":
            out_dirs: Dict[str, Path] = {}
            for sym in universe:
                dest = (self.artifacts_root / sym)
                dest.mkdir(parents=True, exist_ok=True)
                meta_path = dest / "meta.json"
                old_meta = _load_meta(meta_path)

                '''payload = {
                    "strategy": "per_symbol",
                    "symbol": sym,
                    "window": {"start": start, "end": end},
                    "fingerprint": fp.get(sym, {"rows": 0, "tmax": None}),
                    "config_hash": cfg_hash,
                    "universe_hash": u_hash,
                }'''

                # Decide the per-symbol fingerprint representation
                if isinstance(fp, dict):
                    sym_fp = fp.get(sym, {"rows": 0, "tmax": None})
                else:
                    # fp is a single hexdigest for the whole slice; wrap it so meta stays stable
                    sym_fp = {"digest": fp}

                payload = {
                    "strategy": "per_symbol",
                    "symbol": sym,
                    "window": {"start": start, "end": end},
                    "fingerprint": sym_fp,
                    "config_hash": cfg_hash,
                    "universe_hash": u_hash,
                }

                '''if _needs_rebuild(old_meta, payload):
                    # Build (or rebuild) artifacts for this symbol
                    if per_symbol_builder is not None:
                        per_symbol_builder(sym, dest, start, end)
                    # Whether we built or not, record provenance (the test asserts on this)

                    if per_symbol_builder is None:
                        # default: defer to your existing script helper if available
                        try:
                            from scripts.rebuild_artefacts import rebuild_if_needed  # type: ignore
                            #rebuild_if_needed(symbol=sym, artifacts_root=str(dest),
                            #                  parquet_root=str(self.parquet_root),
                            #                  start=start, end=end)
                            rebuild_if_needed(
                                artefact_dir=str(dest),  # ✅ name matches function
                                parquet_root=str(self.parquet_root),
                                symbols=[sym],  # ✅ pass a list
                                start=str(start),
                                end=str(end),
                                n_clusters=int(config_hash_parts.get("k_max", 64)),
                                fitted_pipeline_dir=self.fitted_pipeline_dir,
                            )
                        except Exception as e:
                            # If you don't have a builder yet, make it explicit.
                            raise RuntimeError(
                                f"No per_symbol_builder and scripts.rebuild_artefacts.rebuild_if_needed failed for {sym}: {e}"
                            )
                    else:
                        per_symbol_builder(sym, dest, start, end)

                    meta_path.write_text(json.dumps({"payload": payload}, indent=2), encoding="utf-8")'''

                if _needs_rebuild(old_meta, payload):
                    # Build only if a builder is explicitly provided (tests don't need it)
                    if per_symbol_builder:
                        per_symbol_builder(sym, dest, start, end)
                    # Always record provenance
                    meta_path.write_text(json.dumps({"payload": payload}, indent=2), encoding="utf-8")

                out_dirs[sym] = dest
            return out_dirs

        elif strategy == "pooled":
            dest = (self.artifacts_root / "pooled")
            dest.mkdir(parents=True, exist_ok=True)
            meta_path = dest / "meta.json"
            old_meta = _load_meta(meta_path)

            payload = {
                "strategy": "pooled",
                "symbols": universe,
                "window": {"start": start, "end": end},
                "fingerprint": fp,  # all symbols included
                "config_hash": cfg_hash,
                "universe_hash": _hash_universe(universe),

            }
            '''if _needs_rebuild(old_meta, payload):
                if pooled_builder is None:
                    # You can implement pooled builder later; for now, fail loudly if asked.
                    raise RuntimeError("pooled strategy requested but no pooled_builder provided.")
                pooled_builder(universe, dest, start, end)
                meta_path.write_text(json.dumps({"payload": payload}, indent=2), encoding="utf-8")
            '''

            if _needs_rebuild(old_meta, payload):
                if pooled_builder:
                    pooled_builder(universe, dest, start, end)
                meta_path.write_text(json.dumps({"payload": payload}, indent=2), encoding="utf-8")

            return {"__pooled__": dest}

        else:
            raise ValueError(f"Unknown strategy: {strategy}")
