# prediction_engine/artifacts/manager.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Tuple, Optional
import hashlib, json
import pandas as pd

import pyarrow as pa
import pyarrow.dataset as ds


Strategy = Literal["per_symbol", "pooled"]


def _hash_obj(obj: object) -> str:
    """Stable hash for simple JSON-serializable objects."""
    s = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(s).hexdigest()


def _fingerprint_slice(parquet_root: Path, symbols: Iterable[str], start, end) -> Dict[str, Dict[str, object]]:
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

        if strategy == "per_symbol":
            out_dirs: Dict[str, Path] = {}
            for sym in universe:
                dest = (self.artifacts_root / sym)
                dest.mkdir(parents=True, exist_ok=True)
                meta_path = dest / "meta.json"
                old_meta = _load_meta(meta_path)

                payload = {
                    "strategy": "per_symbol",
                    "symbol": sym,
                    "window": {"start": start, "end": end},
                    "fingerprint": fp.get(sym, {"rows": 0, "tmax": None}),
                    "config_hash": cfg_hash,
                }
                if _needs_rebuild(old_meta, payload):
                    # Build (or rebuild) artifacts for this symbol
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
            }
            if _needs_rebuild(old_meta, payload):
                if pooled_builder is None:
                    # You can implement pooled builder later; for now, fail loudly if asked.
                    raise RuntimeError("pooled strategy requested but no pooled_builder provided.")
                pooled_builder(universe, dest, start, end)
                meta_path.write_text(json.dumps({"payload": payload}, indent=2), encoding="utf-8")

            return {"__pooled__": dest}

        else:
            raise ValueError(f"Unknown strategy: {strategy}")
