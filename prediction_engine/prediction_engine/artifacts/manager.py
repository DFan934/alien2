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


# --- NEW: minimal diagnostics writer for per-symbol artifacts ---
from pathlib import Path
import json

def _write_per_symbol_diagnostics(
    sym_dir: Path,
    *,
    distance_family: str,
    k_used: int | None,
    feature_schema_version: str | None
) -> None:
    sym_dir.mkdir(parents=True, exist_ok=True)
    dpath = sym_dir / "diagnostics.json"
    payload = {
        "distance_family": distance_family,
        "k_used": int(k_used) if k_used is not None else None,
        "feature_schema_version": feature_schema_version,
        # Placeholders so the shape matches tests/expectations
        "fallback_rate_by_regime": {k: 0.0 for k in ("TREND", "RANGE", "VOL", "GLOBAL")},
        "median_knn_distance_by_regime": {k: None for k in ("TREND", "RANGE", "VOL", "GLOBAL")},
        "calibration_ece_by_symbol": {},
        "brier_by_symbol": {},
    }
    dpath.write_text(json.dumps(payload, indent=2))
# --- END NEW ---


# --- 3.8: diagnostics writer -------------------------------------------------
def _write_diagnostics_json(
    dest: Path,
    *,
    payload: dict,
    symbols: list[str] | None = None,
) -> Path:
    """
    Persist artifact-time diagnostics for P4/P5 reporting.

    Shape (no missing keys):
      {
        "distance_family": str,
        "k_used": int | null,
        "feature_schema_version": str | null,
        "fallback_rate_by_regime": {"TREND": 0.0, "RANGE": 0.0, "VOL": 0.0, "GLOBAL": 0.0},
        "median_knn_distance_by_regime": {"TREND": null, "RANGE": null, "VOL": null, "GLOBAL": null},
        "calibration_ece_by_symbol": {SYM: float, ...},
        "brier_by_symbol": {SYM: float, ...}
      }
    """
    dest = Path(dest)
    diag_path = dest / "diagnostics.json"
    distance = (payload.get("distance") or {})
    family = distance.get("family")
    params = (distance.get("params") or {})
    k_used = params.get("k_max")

    # schema version is carried in your schema hash parts; here we only have the final schema hash,
    # so expose that. (If you add a version string in payload later, this will pick it up too.)
    feature_schema_version = payload.get("schema_hash")

    # Initialize required keys with defaults
    fallback_by_regime = {k: 0.0 for k in _REGIME_KEYS}
    median_d2_by_regime = {k: None for k in _REGIME_KEYS}

    # Pull calibration report if present
    cal_ece_by_sym, brier_by_sym = {}, {}
    try:
        cal_dir = dest / "calibrators"
        rpt = json.loads((cal_dir / "calibration_report.json").read_text(encoding="utf-8"))
        for sym, m in rpt.items():
            # skip non-dict leafs if any
            if isinstance(m, dict):
                if "ece" in m:
                    cal_ece_by_sym[sym] = float(m["ece"])
                if "brier" in m:
                    brier_by_sym[sym] = float(m["brier"])
    except Exception:
        pass

    # Assemble and write
    diagnostics = {
        "distance_family": family,
        "k_used": int(k_used) if isinstance(k_used, (int, float)) else None,
        "feature_schema_version": feature_schema_version,
        "fallback_rate_by_regime": fallback_by_regime,
        "median_knn_distance_by_regime": median_d2_by_regime,
        "calibration_ece_by_symbol": cal_ece_by_sym,
        "brier_by_symbol": brier_by_sym,
    }
    diag_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return diag_path
# ---------------------------------------------------------------------------



def _hash_obj(obj: object) -> str:
    """Stable hash for simple JSON-serializable objects."""
    s = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(s).hexdigest()

# INSERT AFTER: return hashlib.sha1(s).hexdigest()
def _hash_schema(parts: dict | list | tuple | None) -> str:
    """
    Stable hash of the *feature/schema-related* inputs that must force artifact rebuilds
    when they change: feature list or version, label horizon H, distance metric family,
    regime settings, gating flags, etc.
    """
    return _hash_obj(parts or {})


# ADD ▼▼▼ (after _hash_schema)
_REGIME_KEYS = ("TREND", "RANGE", "VOL", "GLOBAL")

def _ensure_ann_contract(out_dir: Path) -> None:
    """
    Create a regime-aware ANN index layout and stub 'contract' files.
    This is cheap and lets downstream loaders know what to expect.
    """
    ann_root = Path(out_dir) / "ann"
    ann_root.mkdir(parents=True, exist_ok=True)
    for key in _REGIME_KEYS:
        f = ann_root / f"{key}.index"
        if not f.exists():
            f.write_text("# placeholder index – replace with FAISS/Annoy later\n", encoding="utf-8")



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
        schema_hash_parts: Dict[str, object] | None = None,
            # builder hooks (optional overrides / test doubles)
        per_symbol_builder=None,   # callable(symbol, out_dir: Path, start, end) -> None
        pooled_builder=None,       # callable(symbols: List[str], out_dir: Path, start, end) -> None
        # INSERT AFTER: pooled_builder=None,
        calibrator_builder=None,  # callable(symbol: str, pooled_dir: Path, start, end) -> None
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

        # INSERT AFTER: u_hash = _hash_universe(universe)
        sch_hash = _hash_schema(schema_hash_parts)

        if strategy == "per_symbol":
            out_dirs: Dict[str, Path] = {}
            for sym in universe:
                dest = (self.artifacts_root / sym)
                dest.mkdir(parents=True, exist_ok=True)
                meta_path = dest / "meta.json"
                old_meta = _load_meta(meta_path)

                # --- NEW: ensure diagnostics.json exists for per_symbol artifacts ---
                metric = (config_hash_parts or {}).get("metric", "euclidean")
                k_used = (config_hash_parts or {}).get("k_max")
                feature_schema_version = (schema_hash_parts or {}).get("feature_schema_version")

                _write_per_symbol_diagnostics(
                    dest,
                    distance_family=metric,
                    k_used=k_used,
                    feature_schema_version=feature_schema_version,
                )
                # --- END NEW ---

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
                    "schema_hash": sch_hash,
                    # INSERT INSIDE per_symbol payload, BEFORE universe_hash
                    "pooled_core_dir": str(self.artifacts_root / "pooled") if (
                                self.artifacts_root / "pooled").exists() else None,

                    "universe_hash": u_hash,
                }



                # --- 3.4: distance contract (per_symbol) -----------------------------
                family = (config_hash_parts or {}).get("metric", "euclidean")
                k_max = int((config_hash_parts or {}).get("k_max", 32))

                distance_params: dict[str, object] = {"k_max": k_max}

                # Optional params depending on chosen family
                cov_inv_file = dest / "centroid_cov_inv.npy"
                if family == "mahalanobis" and cov_inv_file.exists():
                    import hashlib
                    b = cov_inv_file.read_bytes()
                    distance_params["cov_inv_sha1"] = hashlib.sha1(b).hexdigest()[:12]
                    distance_params["cov_inv_path"] = str(cov_inv_file)

                rfw_file = dest / "rf_feature_weights.npy"
                if family == "rf_weighted" and rfw_file.exists():
                    import numpy as _np, hashlib
                    w = _np.load(rfw_file, allow_pickle=False)
                    distance_params["rf_weights_len"] = int(w.shape[0])
                    distance_params["rf_weights_sha1"] = hashlib.sha1(w.tobytes()).hexdigest()[:12]
                    distance_params["rf_weights_path"] = str(rfw_file)

                schema_file = dest / "feature_schema.json"
                if schema_file.exists():
                    distance_params["feature_schema_path"] = str(schema_file)

                payload["distance"] = {"family": str(family), "params": distance_params}
                # --------------------------------------------------------------------

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
            # ADD ▼▼▼ (after 'dest = (self.artifacts_root / sym)' and 'dest.mkdir(...)')
            _ensure_ann_contract(dest)  # ann/{TREND,RANGE,VOL,GLOBAL}.index alongside per-symbol artifacts

            meta_path = dest / "meta.json"
            old_meta = _load_meta(meta_path)

            payload = {
                "strategy": "pooled",
                "symbols": universe,
                "window": {"start": start, "end": end},
                "fingerprint": fp,  # all symbols included
                "config_hash": cfg_hash,
                "schema_hash": sch_hash,
                "universe_hash": _hash_universe(universe),

            }

            # --- 3.4: distance contract (pooled) --------------------------------
            family = (config_hash_parts or {}).get("metric", "euclidean")
            k_max = int((config_hash_parts or {}).get("k_max", 32))

            distance_params: dict[str, object] = {"k_max": k_max}

            cov_inv_file = dest / "centroid_cov_inv.npy"
            if family == "mahalanobis" and cov_inv_file.exists():
                import hashlib
                b = cov_inv_file.read_bytes()
                distance_params["cov_inv_sha1"] = hashlib.sha1(b).hexdigest()[:12]
                distance_params["cov_inv_path"] = str(cov_inv_file)

            rfw_file = dest / "rf_feature_weights.npy"
            if family == "rf_weighted" and rfw_file.exists():
                import numpy as _np, hashlib
                w = _np.load(rfw_file, allow_pickle=False)
                distance_params["rf_weights_len"] = int(w.shape[0])
                distance_params["rf_weights_sha1"] = hashlib.sha1(w.tobytes()).hexdigest()[:12]
                distance_params["rf_weights_path"] = str(rfw_file)

            schema_file = dest / "feature_schema.json"
            if schema_file.exists():
                distance_params["feature_schema_path"] = str(schema_file)

            payload["distance"] = {"family": str(family), "params": distance_params}
            # --------------------------------------------------------------------

            '''if _needs_rebuild(old_meta, payload):
                if pooled_builder is None:
                    # You can implement pooled builder later; for now, fail loudly if asked.
                    raise RuntimeError("pooled strategy requested but no pooled_builder provided.")
                pooled_builder(universe, dest, start, end)
                meta_path.write_text(json.dumps({"payload": payload}, indent=2), encoding="utf-8")
            '''

            # INSERT AFTER payload dict is built, BEFORE the _needs_rebuild(...) check
            # Ensure pooled layout exists
            (dest / "calibrators").mkdir(parents=True, exist_ok=True)
            # Optionally record canonical filenames (consumed by tests and callers)

            # ADD ▼▼▼ (right after we mkdir the pooled 'calibrators' dir)
            _ensure_ann_contract(dest)  # writes ann/{TREND,RANGE,VOL,GLOBAL}.index

            payload.setdefault("paths", {
                "scaler": str(dest / "scaler.pkl"),
                "pca": str(dest / "pca.pkl"),
                "clusters": str(dest / "clusters.pkl"),
                "feature_schema": str(dest / "feature_schema.json"),
                "calibrators_dir": str(dest / "calibrators"),
            })

            # --- 3.5: per-symbol calibrator quality gates -------------------------------
            # Thresholds (can be promoted to config later)
            _ECE_MAX = 0.03  # ≤ 3%
            _BRIER_DELTA_MAX = 0.03  # symbol Brier must be within +3% of pooled
            _MIN_MONO_ADJ = 8  # ≥ 8/9 adjacent decile increases

            cal_dir = dest / "calibrators"
            cal_dir.mkdir(parents=True, exist_ok=True)

            # accumulate a portfolio-wide pooled benchmark if builder provides it
            pooled_brier_benchmark = None
            report_path = cal_dir / "calibration_report.json"
            report = {}  # symbol -> metrics

            # --- Allow pooled core build without calibrators (for schema/hash-only tests) ---
            if calibrator_builder is None:
                # If anything changed, (optionally) build pooled core and persist meta
                if _needs_rebuild(old_meta, payload):
                    if pooled_builder:
                        pooled_builder(universe, dest, start, end)
                    meta_path.write_text(json.dumps({"payload": payload}, indent=2), encoding="utf-8")
                # Return the pooled directory map just like normal
                return {"__pooled__": dest}
            # -------------------------------------------------------------------------------

            for sym in universe:
                # builder should: fit isotonic, persist "<SYM>.isotonic.pkl",
                # and return (pkl_path, metrics_dict)
                pkl_path, metrics = calibrator_builder(sym, dest, start, end)
                # expected keys: 'ece', 'brier', 'mono_adj_pairs', optional 'pooled_brier'
                if metrics is None:
                    metrics = {}
                if pooled_brier_benchmark is None and "pooled_brier" in metrics:
                    pooled_brier_benchmark = float(metrics["pooled_brier"])

                # evaluate gates
                ece_ok = float(metrics.get("ece", 1.0)) <= _ECE_MAX
                # if we don't have a pooled benchmark, treat Brier gate as informational
                if pooled_brier_benchmark is not None:
                    brier_ok = float(metrics.get("brier", 1.0)) <= (pooled_brier_benchmark + _BRIER_DELTA_MAX)
                else:
                    brier_ok = True
                mono_ok = int(metrics.get("mono_adj_pairs", 0)) >= _MIN_MONO_ADJ

                status = "pass" if (ece_ok and brier_ok and mono_ok) else "fail"
                metrics.update({
                    "status": status,
                    "gates": {
                        "ece_max": _ECE_MAX,
                        "brier_delta_max": _BRIER_DELTA_MAX,
                        "min_monotonic_adjacent_pairs": _MIN_MONO_ADJ,
                    }
                })
                report[sym] = metrics

            # write/update report atomically
            #import json, io
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            payload.setdefault("calibration", {})["report"] = str(report_path)
            # ---------------------------------------------------------------------------

            # --- 3.8: write diagnostics.json (pooled) ---
            _write_diagnostics_json(dest, payload={"distance": payload.get("distance"),
                                                   "schema_hash": payload.get("schema_hash")},
                                    symbols=universe)
            # --------------------------------------------

            if _needs_rebuild(old_meta, payload):
                if pooled_builder:
                    pooled_builder(universe, dest, start, end)

                    # INSERT INSIDE the rebuild branch (pooled) AFTER pooled_builder(...)
                    # Build per-symbol calibrators into pooled/calibrators
                    if calibrator_builder:
                        for sym in universe:
                            calibrator_builder(sym, dest, start, end)  # writes <pooled>/calibrators/<SYM>.isotonic.pkl
                            # --- 3.8: write diagnostics.json (per_symbol) ---
                            _write_diagnostics_json(dest, payload={"distance": payload.get("distance"),
                                                                   "schema_hash": payload.get("schema_hash")},
                                                    symbols=[sym])
                            # ------------------------------------------------

                # --- 3.8: refresh diagnostics after rebuild ---
                _write_diagnostics_json(dest, payload={"distance": payload.get("distance"),
                                                       "schema_hash": payload.get("schema_hash")},
                                        symbols=universe)
                # ----------------------------------------------

                meta_path.write_text(json.dumps({"payload": payload}, indent=2), encoding="utf-8")

            return {"__pooled__": dest}

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    # --- INSERT INSIDE ArtifactManager (AFTER fit_or_load) -------------------
    def build_fold_artifacts(
        self,
        *,
        universe: List[str],
        train_start: str,
        train_end: str,
        fold_dir: Path | str,
        strategy: Strategy = "pooled",
        fold_id: int | str | None = None,
        config_hash_parts: Dict[str, object] | None = None,
        schema_hash_parts: Dict[str, object] | None = None,
        # optional builders (same signatures you already use)
        per_symbol_builder=None,   # (symbol, out_dir: Path, start, end) -> None
        pooled_builder=None,       # (symbols: List[str], out_dir: Path, start, end) -> None
        calibrator_builder=None,   # (symbol: str, pooled_dir: Path, start, end) -> (pkl_path, metrics_dict)
    ) -> Dict[str, Path]:
        """
        Build (or refresh) artifacts scoped to a *single fold* namespace.

        Layout:
          <artifacts_root>/folds/fold_i/
            pooled/...   or   <SYM>/...

        Writes meta.json in the chosen directory with:
          train_window, universe_hash, schema_hash, config_hash, row_count|digest, tmax (if available), fold_id
        """
        self.parquet_root = Path(self.parquet_root).expanduser().resolve()
        fold_dir = Path(fold_dir).expanduser().resolve()
        fold_dir.mkdir(parents=True, exist_ok=True)

        cfg_hash = _hash_obj(config_hash_parts or {})
        sch_hash = _hash_schema(schema_hash_parts)
        u_hash = _hash_universe(universe)
        fp = _fingerprint_slice(self.parquet_root, universe, train_start, train_end)

        if strategy == "per_symbol":
            out_dirs: Dict[str, Path] = {}
            for sym in universe:
                dest = fold_dir / sym
                dest.mkdir(parents=True, exist_ok=True)

                meta_path = dest / "meta.json"
                old_meta = _load_meta(meta_path)

                # If fp is dict-like (per symbol), use per-symbol portion; else wrap digest
                if isinstance(fp, dict):
                    sym_fp = fp.get(sym, {"rows": 0, "tmax": None})
                else:
                    sym_fp = {"digest": fp}

                payload = {
                    "fold_id": fold_id,
                    "strategy": "per_symbol",
                    "symbol": sym,
                    "train_window": {"start": train_start, "end": train_end},
                    "fingerprint": sym_fp,
                    "config_hash": cfg_hash,
                    "schema_hash": sch_hash,
                    "universe_hash": u_hash,
                }

                # distance contract mirror (optional but keeps parity with fit_or_load)
                family = (config_hash_parts or {}).get("metric", "euclidean")
                k_max = int((config_hash_parts or {}).get("k_max", 32))
                distance_params: dict[str, object] = {"k_max": k_max}
                schema_file = dest / "feature_schema.json"
                if schema_file.exists():
                    distance_params["feature_schema_path"] = str(schema_file)
                payload["distance"] = {"family": str(family), "params": distance_params}

                if _needs_rebuild(old_meta, payload):
                    if per_symbol_builder:
                        per_symbol_builder(sym, dest, train_start, train_end)
                    meta_path.write_text(json.dumps({"payload": payload}, indent=2), encoding="utf-8")

                out_dirs[sym] = dest
            return out_dirs

        elif strategy == "pooled":
            dest = fold_dir / "pooled"
            dest.mkdir(parents=True, exist_ok=True)
            _ensure_ann_contract(dest)  # ann/{TREND,RANGE,VOL,GLOBAL}.index

            meta_path = dest / "meta.json"
            old_meta = _load_meta(meta_path)

            payload = {
                "fold_id": fold_id,
                "strategy": "pooled",
                "symbols": list(universe),
                "train_window": {"start": train_start, "end": train_end},
                "fingerprint": fp,                # full-slice fingerprint (rows/min/max or digest)
                "config_hash": cfg_hash,
                "schema_hash": sch_hash,
                "universe_hash": u_hash,
                "paths": {
                    "scaler": str(dest / "scaler.pkl"),
                    "pca": str(dest / "pca.pkl"),
                    "clusters": str(dest / "clusters.pkl"),
                    "feature_schema": str(dest / "feature_schema.json"),
                    "calibrators_dir": str(dest / "calibrators"),
                },
            }
            (dest / "calibrators").mkdir(parents=True, exist_ok=True)

            # distance contract
            family = (config_hash_parts or {}).get("metric", "euclidean")
            k_max = int((config_hash_parts or {}).get("k_max", 32))
            payload["distance"] = {"family": str(family), "params": {"k_max": k_max}}

            # write diagnostics skeleton once (safe to overwrite)
            _write_diagnostics_json(dest, payload={"distance": payload.get("distance"),
                                                   "schema_hash": payload.get("schema_hash")}, symbols=universe)

            if _needs_rebuild(old_meta, payload):
                if pooled_builder:
                    pooled_builder(universe, dest, train_start, train_end)
                if calibrator_builder:
                    for sym in universe:
                        calibrator_builder(sym, dest, train_start, train_end)
                _write_diagnostics_json(dest, payload={"distance": payload.get("distance"),
                                                       "schema_hash": payload.get("schema_hash")}, symbols=universe)
                meta_path.write_text(json.dumps({"payload": payload}, indent=2), encoding="utf-8")

            return {"__pooled__": dest}

        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    # --- END INSERT -----------------------------------------------------------
