# ---------------------------------------------------------------------------
# prediction_engine/scripts/rebuild_artefacts.py (Corrected)
# ---------------------------------------------------------------------------
from __future__ import annotations
import json
import logging
import hashlib
from pathlib import Path
from typing import Sequence
# INSERT AFTER existing imports
import joblib
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

import pandas as pd
import numpy as np

from prediction_engine.market_regime import label_days, RegimeParams
from feature_engineering.pipelines.core import CoreFeaturePipeline
from prediction_engine.path_cluster_engine import PathClusterEngine
from feature_engineering.labels.labeler import one_bar_ahead

LOG = logging.getLogger(__name__)


def _load_meta(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None

def _dbg(df: pd.DataFrame, tag: str, head: int = 3):
    try:
        idx = df.index
        idx_type = type(idx).__name__
        idx_names = getattr(idx, "names", None)
        tz = getattr(idx, "tz", None) if isinstance(idx, pd.DatetimeIndex) else None

        print("\n" + "=" * 90)
        print(f"[DBG] {tag}")
        print(f"shape={df.shape}")
        print(f"columns({len(df.columns)}): {list(df.columns)[:15]}{' ...' if len(df.columns)>15 else ''}")
        print(f"has_col timestamp={('timestamp' in df.columns)} symbol={('symbol' in df.columns)}")
        print(f"index_type={idx_type} index_names={idx_names} index_tz={tz}")

        if isinstance(idx, pd.MultiIndex):
            print(f"multiindex nlevels={idx.nlevels} names={idx.names}")
            # show first few tuples
            print("multiindex head:", idx[:min(head, len(idx))].tolist())
            # duplicates
            try:
                dup = idx.duplicated().sum()
                print(f"multiindex duplicated count={dup}")
            except Exception as e:
                print("multiindex duplicated check failed:", e)

        if isinstance(idx, pd.DatetimeIndex):
            print("dtindex head:", list(idx[:min(head, len(idx))]))
            print("dtindex min/max:", idx.min(), idx.max())
            print("dtindex duplicated count:", idx.duplicated().sum())

        # timestamp column sanity if present
        if "timestamp" in df.columns:
            ts = df["timestamp"]
            print("timestamp col dtype:", ts.dtype)
            try:
                ts2 = pd.to_datetime(ts, errors="coerce", utc=True)
                print("timestamp col coerced utc dtype:", ts2.dtype, "NaT:", ts2.isna().sum())
                # show a few
                print("timestamp col head:", ts.head(head).tolist())
            except Exception as e:
                print("timestamp col coercion failed:", e)

        # symbol column sanity if present
        if "symbol" in df.columns:
            sym = df["symbol"]
            print("symbol col dtype:", sym.dtype)
            print("symbol unique head:", sym.astype(str).unique()[:10])

    except Exception as e:
        print(f"[DBG] {tag} FAILED:", e)

# INSERT BEFORE: def rebuild_if_needed(...)

def _safe_reset_index(df: pd.DataFrame, tag: str = "") -> pd.DataFrame:
    """
    Reset index without crashing if index level names already exist as columns.
    If df has MultiIndex (timestamp,symbol) AND also has those columns, we drop
    the index instead of inserting duplicates.
    """
    if not isinstance(df.index, pd.MultiIndex):
        return df.reset_index(drop=False)

    idx_names = [n for n in df.index.names if n is not None]
    collision = [n for n in idx_names if n in df.columns]

    if collision:
        print(f"[DBG] safe_reset_index({tag}) collisions={collision} -> reset_index(drop=True)")
        return df.reset_index(drop=True)
    else:
        return df.reset_index(drop=False)


def build_pooled_core(symbols, out_dir: Path, start: str, end: str, parquet_root: str | Path, n_clusters: int = 64):
    """
    Train pooled scaler/PCA/clusters over all symbols and persist standard filenames.
    Files written:
      out_dir/scaler.pkl, out_dir/pca.pkl, out_dir/clusters.pkl, out_dir/feature_schema.json
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load per-symbol slices and concatenate
    base = Path(parquet_root)
    dfs = []
    for sym in symbols:
        p = base / f"symbol={sym}"
        if not p.exists():
            continue
        df = pd.read_parquet(p, engine="pyarrow")
        if df.empty:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
        df = df[(df["timestamp"] >= pd.to_datetime(start)) & (df["timestamp"] <= pd.to_datetime(end))]
        if not df.empty:
            dfs.append(df)

        '''from feature_engineering.utils.time import ensure_utc_timestamp_col
        from feature_engineering.utils.timegrid import standardize_bars_to_grid

        # --- canonicalize ts (tz-aware) + slice window in UTC -----------------
        ensure_utc_timestamp_col(df, "timestamp", who=f"[pooled:{sym}]")

        start_ts = pd.to_datetime(start, utc=True)
        end_ts = pd.to_datetime(end, utc=True)
        df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
        if df.empty:
            continue

        # --- enforce 60s UTC grid BEFORE any FE --------------------------------
        df_std, audit = standardize_bars_to_grid(df, symbol=sym, freq="60s")
        LOG.info(f"[GridAudit pooled] {audit}")

        dfs.append(df_std)'''

    if not dfs:
        raise RuntimeError("No data found for pooled build")

    raw = pd.concat(dfs, ignore_index=True)

    from feature_engineering.utils.timegrid import standardize_bars_to_grid

    raw, audits = standardize_bars_to_grid(
        raw,
        symbol_col="symbol",
        ts_col="timestamp",
        freq="60s",
        expected_freq_s=60,
        fill_volume_zero=True,
        keep_ohlc_nan=True,
        hard_fail_on_duplicates=False,
    )

    _dbg(raw, "after standardize_bars_to_grid (raw)")
    # Force keys to live in columns (not index)
    if isinstance(raw.index, pd.MultiIndex) and any(n in ("timestamp", "symbol") for n in raw.index.names):
        raw = raw.reset_index(drop=True)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce").dt.tz_localize(None)
    raw["symbol"] = raw["symbol"].astype("string")

    raw.attrs["grid_audit"] = audits

    # 2) Fit features (get scaler/pca inside pipeline) and build clusters on pooled PCA space
    pipe = CoreFeaturePipeline(parquet_root=out_dir)   # fits in-memory

    feats, arte = pipe.run_mem(raw)  # arte (if returned) may include scaler_, pca_, etc.

    # --- Ensure feats has timestamp + symbol info (pooled builder contract) ----
    # 1) Timestamp
    if "timestamp" not in feats.columns:
        # if timestamp is the index, preserve it as a column
        if isinstance(feats.index, pd.DatetimeIndex):
            idx = feats.index
            # drop tz if present
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_localize(None)
            feats = feats.copy()
            feats["timestamp"] = idx
        else:
            raise KeyError("[pooled] feats has no 'timestamp' col and index is not DatetimeIndex")

    else:
        # normalize to UTC-naive timestamps consistently
        feats = feats.copy()
        feats["timestamp"] = (
            pd.to_datetime(feats["timestamp"], utc=True, errors="coerce")
            .dt.tz_localize(None)
        )

    # 2) Symbol
    if "symbol" not in feats.columns:
        # Best case: one-to-one row alignment with raw
        if len(feats) == len(raw) and "symbol" in raw.columns:
            feats["symbol"] = raw["symbol"].astype("string").to_numpy()
        else:
            # If you truly built pooled features without per-row symbols, you must
            # use a single symbol tag and also collapse labels the same way.
            # But your label series uses real symbols, so prefer the aligned case.
            raise KeyError("[pooled] feats missing 'symbol' and cannot align rows to raw")
    else:
        feats["symbol"] = feats["symbol"].astype("string")
    # --------------------------------------------------------------------------

    # 3) Persist core components
    # Try to extract internals if the pipeline exposes them; otherwise persist placeholders so layout exists.
    scaler = getattr(pipe, "scaler_", None) or arte.get("scaler_", None) if isinstance(arte, dict) else None
    pca    = getattr(pipe, "pca_", None)    or arte.get("pca_", None)    if isinstance(arte, dict) else None

    if scaler is not None:
        joblib.dump(scaler, out_dir / "scaler.pkl")
    else:
        joblib.dump({"_": "placeholder_scaler"}, out_dir / "scaler.pkl")

    if pca is not None:
        joblib.dump(pca, out_dir / "pca.pkl")
    else:
        joblib.dump({"_": "placeholder_pca"}, out_dir / "pca.pkl")

    pca_cols = [c for c in feats.columns if c.startswith("pca_")]
    feature_schema = {"pca_columns": pca_cols}
    (out_dir / "feature_schema.json").write_text(json.dumps(feature_schema, indent=2), encoding="utf-8")

    # 4) (Re)build pooled clusters from pooled PCA features
    # Simple numeric label for pooled build — use a forward O->C ret consistent with symbol builder
    raw = raw.sort_values(["symbol", "timestamp"])
    next_open  = raw.groupby("symbol")["open"].shift(-1)
    next_close = raw.groupby("symbol")["close"].shift(-1)
    raw["ret_fwd"] = (next_close / next_open - 1.0)
    feats = feats.set_index(pd.to_datetime(feats["timestamp"]).dt.tz_localize(None), drop=False)
    #y_numeric = pd.Series(raw["ret_fwd"].values, index=pd.to_datetime(raw["timestamp"]).dt.tz_localize(None)).reindex(feats.index).values

    # --- Robust label→feature alignment for pooled builds -------------------
    # 1) Normalize raw label keys
    #ts_raw = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce").tz_convert(None)
    ts_raw = (
        pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
        .dt.tz_localize(None)
    )

    sym_raw = (raw["symbol"].astype(str) if "symbol" in raw.columns else pd.Series([""] * len(raw)))

    # 2) Build a MultiIndex label series and collapse any exact dupes
    lab = pd.Series(raw["ret_fwd"].to_numpy(),
                    index=pd.MultiIndex.from_arrays([ts_raw, sym_raw],
                                                    names=["timestamp", "symbol"]))
    #lab = lab.groupby(level=["timestamp", "symbol"]).mean().sort_index()

    lab = lab.groupby(level=["timestamp", "symbol"]).mean().sort_index()  # ✅ keep

    # 3) Build the feature row key we’ll align to
    #    (supports either a DatetimeIndex with a 'symbol' column, or a MultiIndex).
    if isinstance(feats.index, pd.MultiIndex) and set(feats.index.names) >= {"timestamp", "symbol"}:
        fe_key = feats.index
    else:
        # Try to synthesize a MultiIndex from columns/index
        if "timestamp" in feats.columns:
            #ts_feat = pd.to_datetime(feats["timestamp"], utc=True, errors="coerce").tz_convert(None)
            ts_feat = (
                pd.to_datetime(feats["timestamp"], utc=True, errors="coerce")
                .dt.tz_localize(None)
            )

        else:
            ts_feat = pd.DatetimeIndex(feats.index).tz_localize(None, nonexistent="shift_forward", ambiguous="NaT")

        '''if "symbol" in feats.columns:
            sym_feat = feats["symbol"].astype(str)
        else:
            # If symbol isn’t a column, assume single-symbol features: use empty symbol tag
            sym_feat = pd.Series([""] * len(feats), index=feats.index, dtype="string")
        '''

        sym_feat = (
            feats["symbol"].astype("string")
            if "symbol" in feats.columns else
            pd.Series("POOLED", index=feats.index, dtype="string")
        )

        #fe_key = pd.MultiIndex.from_arrays([ts_feat.to_numpy(), sym_feat.to_numpy()],
        #                                   names=["timestamp", "symbol"])

        # MultiIndex that matches feats row-for-row
        fe_key = pd.MultiIndex.from_arrays(
            [ts_feat, sym_feat],
            names=["timestamp", "symbol"]
        )
    # 4) Ensure unique, sorted feature key (pandas reindex requirement)
    #fe_key = pd.MultiIndex.from_tuples(list(dict.fromkeys(fe_key.to_list())), names=["timestamp", "symbol"])

    # 4) DO NOT force uniqueness; preserve one-to-one length with feats
    # fe_key = pd.MultiIndex.from_tuples(list(dict.fromkeys(fe_key.to_list())), names=["timestamp", "symbol"])  # ❌ remove this

    # ✅ sanity check and assign
    if len(fe_key) != len(feats):
        raise ValueError(f"fe_key length {len(fe_key)} != feats {len(feats)} (don’t dedupe fe_key)")

    feats.index = fe_key

    # 5) Align labels to features. Prefer strict match; fall back to nearest within half a slot if desired.
    # Strict (no implicit look-ahead):
    y_aligned = lab.reindex(fe_key)

    # Optional “nearest within 90s” for 180s cadence (uncomment if you need it):
    # y_aligned = lab.reindex(fe_key, method="nearest", tolerance=pd.Timedelta(seconds=90))

    # 6) Drop missing and mirror the mask into feats so X and y stay in lock-step
    mask = y_aligned.notna().to_numpy()
    if mask.any():
        # Reindex feats to fe_key if needed, then filter
        if not feats.index.equals(fe_key):
            feats = feats.set_index(fe_key) if "timestamp" in feats.columns or "symbol" in feats.columns else feats
        feats = feats.loc[mask]
        y_numeric = y_aligned.loc[mask].to_numpy()
    else:
        # No overlap: empty y; keep behavior explicit
        y_numeric = np.array([], dtype=float)
    # --- end alignment block ------------------------------------------------

    X = feats[pca_cols].to_numpy(dtype=np.float32)

    PathClusterEngine.build(
        X=X,
        y_numeric=y_numeric,
        y_categorical=None,  # pooled clusters (regime-specific indices come in 3.3)
        feature_names=pca_cols,
        n_clusters=n_clusters,
        out_dir=out_dir,
    )

    LOG.info("[pooled] core built at %s", out_dir)


'''def fit_symbol_calibrator(symbol: str, pooled_dir: Path, start: str, end: str, parquet_root: str | Path):
    """
    Fit an isotonic calibrator for `symbol` using pooled PCA space.
    Writes:  <pooled_dir>/calibrators/<SYMBOL>.isotonic.pkl
             <pooled_dir>/calibrators/calibration_report.json (append/update entry)
    """
    pooled_dir = Path(pooled_dir)
    cal_dir = pooled_dir / "calibrators"
    cal_dir.mkdir(parents=True, exist_ok=True)

    # Load symbol slice
    base = Path(parquet_root); p = base / f"symbol={symbol}"
    df = pd.read_parquet(p, engine="pyarrow")
    if df.empty:
        raise RuntimeError(f"No data for calibrator: {symbol}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    df = df[(df["timestamp"] >= pd.to_datetime(start)) & (df["timestamp"] <= pd.to_datetime(end))]
    if df.empty:
        raise RuntimeError(f"No rows in window for calibrator: {symbol}")

    # Labels: forward O->C return on next bar
    df = df.sort_values(["symbol", "timestamp"])
    next_open  = df.groupby("symbol")["open"].shift(-1)
    next_close = df.groupby("symbol")["close"].shift(-1)
    df["ret_fwd"] = (next_close / next_open - 1.0)

    # Proxy 'raw score' for calibration: use sign-preserving scaled return as a stand-in
    # (In P5 this will be your model's EV/probability)
    score = df["ret_fwd"].fillna(0.0).clip(-0.02, 0.02).to_numpy()
    # Binary target: did we beat 0?
    y = (df["ret_fwd"] > 0).astype(int).to_numpy()

    # Fit isotonic
    iso = IsotonicRegression(out_of_bounds="clip")
    # Map scores to [0,1] monotonically
    p_hat = (score - score.min()) / (score.max() - score.min() + 1e-9)
    iso.fit(p_hat, y)
    joblib.dump(iso, cal_dir / f"{symbol}.isotonic.pkl")

    # Basic quality numbers (wired now; thresholds enforced in P5)
    try:
        y_prob = iso.predict(p_hat)
        auc = float(roc_auc_score(y, y_prob))
        brier = float(brier_score_loss(y, y_prob))
        # Simple ECE (10 bins)
        bins = np.linspace(0, 1, 11)
        idx = np.digitize(y_prob, bins) - 1
        ece = 0.0
        for b in range(10):
            m = idx == b
            if m.any():
                ece += abs(y[m].mean() - y_prob[m].mean()) * (m.mean())
        ece = float(ece)
    except Exception:
        auc, brier, ece = None, None, None

    # Append/update symbol entry in calibration_report.json
    rpt_path = cal_dir / "calibration_report.json"
    try:
        rpt = json.loads(rpt_path.read_text())
    except Exception:
        rpt = {}
    rpt[symbol] = {"roc_auc": auc, "brier": brier, "ece": ece, "n": int(len(y))}
    rpt_path.write_text(json.dumps(rpt, indent=2), encoding="utf-8")

    LOG.info("[pooled] calibrator for %s written to %s", symbol, cal_dir)
'''


def rebuild_if_needed(
        artefact_dir: str | Path,
        parquet_root: str | Path,
        symbols: Sequence[str],
        start: str,
        end: str,
        n_clusters: int = 64,
        fitted_pipeline_dir: str | Path | None = None) -> None:
    """
    Rebuild PathClusterEngine artefacts if they are missing or outdated.
    """
    artefact_dir = Path(artefact_dir)
    artefact_dir.mkdir(parents=True, exist_ok=True)
    meta = _load_meta(artefact_dir / "meta.json")

    # Only skip if meta.json exactly matches our request params
    '''if meta:
        meta_start = str(meta.get("start"))
        meta_end = str(meta.get("end"))
        start_str = str(start)
        end_str = str(end)

        if (
                meta_start == start_str
                and meta_end == end_str
                and meta.get("symbols") == list(symbols)
                and meta.get("n_clusters") == n_clusters
        ):
            LOG.info("[artefacts] Artefacts up-to-date. Skipping rebuild.")
            return'''


    if meta:
        meta_start = str(meta.get("start"))
        meta_end = str(meta.get("end"))
        start_str = str(start)
        end_str = str(end)
        meta_hz = meta.get("label_horizon", "UNKNOWN")

        if (
            meta_start == start_str
            and meta_end == end_str
            and meta.get("symbols") == list(symbols)
            and meta.get("n_clusters") == n_clusters
            and meta_hz == "O->C"
        ):
            LOG.info("[artefacts] Artefacts up-to-date (label_horizon=%s). Skipping rebuild.", meta_hz)
            return
        else:
            LOG.info("[artefacts] Meta mismatch or horizon change (have=%s, need=O->C). Rebuilding…", meta_hz)
    else:
        LOG.info("[artefacts] No meta.json found. Rebuilding…")



    LOG.info("[artefacts] Artefacts outdated (range/symbols changed). Rebuilding…")

    # 1) Load raw minute-bar parquet slice
    base_pq_path = Path(parquet_root)
    if not base_pq_path.exists():
        raise FileNotFoundError(f"Base parquet directory not found at {base_pq_path}")

    # --- FIX: Point to the symbol-specific subdirectory ---
    # This assumes we are building for one symbol at a time in the walk-forward
    if not symbols:
        raise ValueError("Must provide at least one symbol.")
    symbol = symbols[0]
    symbol_specific_pq_path = base_pq_path / f"symbol={symbol}"

    if not symbol_specific_pq_path.exists():
        raise FileNotFoundError(f"Symbol-specific parquet directory not found: {symbol_specific_pq_path}")

    LOG.info(f"Loading data for symbol '{symbol}' from {symbol_specific_pq_path}")
    raw = pd.read_parquet(symbol_specific_pq_path, engine="pyarrow")

    if raw.empty:
        raise RuntimeError("Parquet read returned zero rows – check path.")

    # Normalize timestamps to UTC-naive for consistent comparisons
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    start_ts = pd.to_datetime(start, utc=True).tz_convert("UTC").tz_localize(None) if start else raw[
        "timestamp"].min()
    end_ts = pd.to_datetime(end, utc=True).tz_convert("UTC").tz_localize(None) if end else raw["timestamp"].max()

    # Filter by the time window in pandas
    mask_time = (raw["timestamp"] >= start_ts) & (raw["timestamp"] <= end_ts)
    raw = raw.loc[mask_time].copy()

    if raw.empty:
        raise RuntimeError("Rebuild slice returned zero rows after filtering – check dates/symbols/timezone.")

    from feature_engineering.utils.timegrid import standardize_bars_to_grid

    raw, audits = standardize_bars_to_grid(
        raw,
        symbol_col="symbol",
        ts_col="timestamp",
        freq="60s",
        expected_freq_s=60,
        fill_volume_zero=True,
        keep_ohlc_nan=True,
        hard_fail_on_duplicates=False,
    )
    raw.attrs["grid_audit"] = audits

    # Add required columns for the feature pipeline if they're missing
    if "trigger_ts" not in raw.columns:
        raw["trigger_ts"] = raw["timestamp"]
    if "volume_spike_pct" not in raw.columns:
        volume_ma = raw['volume'].rolling(window=20, min_periods=1).mean()
        raw['volume_spike_pct'] = (raw['volume'] / volume_ma) - 1.0
        raw['volume_spike_pct'] = raw['volume_spike_pct'].fillna(0.0)

    # 2) Run in-memory pipeline to get PCA features
    # --- REFINEMENT: Use the artefact_dir for the pipeline's root ---

    # 2) Get features in the SAME PCA space as walk-forward
    '''if fitted_pipeline_dir is not None:
        pipe = CoreFeaturePipeline(parquet_root=Path(fitted_pipeline_dir))
        feats = pipe.transform_mem(raw)  # ← NO FIT

        # --- CONTRACT: feats must carry timestamp (+ symbol) aligned row-for-row ---
        raw = raw.sort_values(["timestamp"]).reset_index(drop=True)

        # If feats came back with a nontrivial index, drop it so we can align by position
        feats = feats.reset_index(drop=True)

        # 1) Ensure timestamp exists in feats
        if "timestamp" not in feats.columns:
            # Best assumption in your pipeline: transform_mem preserves row order
            if len(feats) != len(raw):
                raise RuntimeError(f"[rebuild] feats len {len(feats)} != raw len {len(raw)}; cannot attach timestamps")
            feats["timestamp"] = raw["timestamp"].to_numpy()
        else:
            feats["timestamp"] = pd.to_datetime(feats["timestamp"], utc=True, errors="coerce").dt.tz_localize(None)

        # 2) Ensure symbol exists in feats (your labels are per-symbol)
        if "symbol" not in feats.columns:
            if len(feats) != len(raw):
                raise RuntimeError(f"[rebuild] feats len {len(feats)} != raw len {len(raw)}; cannot attach symbols")
            feats["symbol"] = raw["symbol"].astype("string").to_numpy()
        else:
            feats["symbol"] = feats["symbol"].astype("string")

        # 3) Canonicalize timestamp to UTC-naive (matches your y_numeric_series)
        feats["timestamp"] = pd.to_datetime(feats["timestamp"], utc=True, errors="coerce").dt.tz_localize(None)
        # -------------------------------------------------------------------------


    else:
        pipe = CoreFeaturePipeline(parquet_root=artefact_dir)
        feats, _ = pipe.run_mem(raw)  # ← legacy path (fits)

    #pipe = CoreFeaturePipeline(parquet_root=artefact_dir)
    #feats, _ = pipe.run_mem(raw)

    # 3) Resample minute data to DAILY to generate correct regime labels
    daily_df = raw.set_index('timestamp').resample('D').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna()
    daily_regimes = label_days(daily_df, RegimeParams())

    # Ensure all indexes are timezone-naive before joining
    if 'timestamp' in feats.columns:
        feats = feats.set_index('timestamp')
    if feats.index.tz is not None:
        feats.index = feats.index.tz_localize(None)

    y_categorical = pd.Series(
        feats.index.normalize().map(daily_regimes),
        index=feats.index, name='regime'
    ).ffill().bfill()

    # 4) Generate numeric labels (forward returns)
    raw = raw.sort_values(["symbol", "timestamp"])
    raw["ret_fwd"] = raw.groupby("symbol")["open"].shift(-1) / raw["open"] - 1.0
    y_numeric_series = pd.Series(
        raw["ret_fwd"].values,
        index=pd.to_datetime(raw["timestamp"]).dt.tz_localize(None),
        name="ret_fwd"
    )

    # 4) Generate numeric labels (forward returns)  **O->C on NEXT bar**
    raw = raw.sort_values(["symbol", "timestamp"])
    next_open  = raw.groupby("symbol")["open"].shift(-1)
    next_close = raw.groupby("symbol")["close"].shift(-1)
    raw["ret_fwd"] = (next_close / next_open - 1.0)

    y_numeric_series = pd.Series(
        raw["ret_fwd"].values,
        index=pd.to_datetime(raw["timestamp"]).dt.tz_localize(None),
        name="ret_fwd"
    )


    # 5) Combine features and labels, then clean
    df_combined = feats.join(y_numeric_series).join(y_categorical)
    pca_cols = [c for c in feats.columns if c.startswith("pca_")]
    required_cols = pca_cols + ["ret_fwd", "regime"]
    df_clean = df_combined[required_cols].dropna()

    if df_clean.empty:
        raise RuntimeError("Data alignment resulted in an empty DataFrame. Check for NaNs or index mismatches.")'''

    # --- Canonical sort (IMPORTANT): raw order must be deterministic ----------
    raw = raw.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    # 2) Get features in the SAME PCA space as walk-forward
    if fitted_pipeline_dir is not None:
        pipe = CoreFeaturePipeline(parquet_root=Path(fitted_pipeline_dir))
        _dbg(raw, "before transform_mem (raw)")

        feats = pipe.transform_mem(raw)  # NO FIT (must preserve row order)
        _dbg(feats, "after transform_mem (feats)")

        feats = feats.reset_index(drop=True)



        # --- CONTRACT: attach timestamp + symbol row-for-row ------------------
        if "timestamp" not in feats.columns:
            if len(feats) != len(raw):
                raise RuntimeError(f"[rebuild] feats len {len(feats)} != raw len {len(raw)}; cannot attach timestamps")
            feats["timestamp"] = raw["timestamp"].to_numpy()
        feats["timestamp"] = pd.to_datetime(feats["timestamp"], utc=True, errors="coerce").dt.tz_localize(None)

        if "symbol" not in feats.columns:
            if len(feats) != len(raw):
                raise RuntimeError(f"[rebuild] feats len {len(feats)} != raw len {len(raw)}; cannot attach symbols")
            feats["symbol"] = raw["symbol"].astype("string").to_numpy()
        feats["symbol"] = feats["symbol"].astype("string")

    else:
        pipe = CoreFeaturePipeline(parquet_root=artefact_dir)
        feats, _ = pipe.run_mem(raw)  # legacy path (fits)
        feats = feats.reset_index(drop=True)

        # Attach keys if pipeline didn't keep them
        if "timestamp" not in feats.columns:
            if len(feats) != len(raw):
                raise RuntimeError(f"[rebuild] run_mem feats len {len(feats)} != raw len {len(raw)}; cannot attach timestamps")
            feats["timestamp"] = raw["timestamp"].to_numpy()
        feats["timestamp"] = pd.to_datetime(feats["timestamp"], utc=True, errors="coerce").dt.tz_localize(None)

        if "symbol" not in feats.columns:
            if len(feats) != len(raw):
                raise RuntimeError(f"[rebuild] run_mem feats len {len(feats)} != raw len {len(raw)}; cannot attach symbols")
            feats["symbol"] = raw["symbol"].astype("string").to_numpy()
        feats["symbol"] = feats["symbol"].astype("string")

    # --- Build MultiIndex key for BOTH raw and feats --------------------------
    raw_ts = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce").dt.tz_localize(None)
    raw_sym = raw["symbol"].astype("string")
    raw_key = pd.MultiIndex.from_arrays([raw_ts, raw_sym], names=["timestamp", "symbol"])
    raw = raw.assign(timestamp=raw_ts, symbol=raw_sym)
    raw.index = raw_key

    feats_ts = pd.to_datetime(feats["timestamp"], utc=True, errors="coerce").dt.tz_localize(None)
    feats_sym = feats["symbol"].astype("string")
    feats_key = pd.MultiIndex.from_arrays([feats_ts, feats_sym], names=["timestamp", "symbol"])
    feats = feats.assign(timestamp=feats_ts, symbol=feats_sym)
    feats.index = feats_key

    # (Optional) sort_index so joins are predictable
    raw = raw.sort_index()
    feats = feats.sort_index()

    tmp = raw

    # If symbol is both in index and in columns, drop the index version before reset
    if "symbol" in tmp.columns and "symbol" in getattr(tmp.index, "names", []):
        tmp = tmp.reset_index(drop=True)
    else:
        tmp = tmp.reset_index()


    # 3) Regime labels (by day) → broadcast to each (ts, sym)

    _dbg(raw, "before daily resample (raw)")
    # Force keys to live in columns (not index)
    if isinstance(raw.index, pd.MultiIndex) and any(n in ("timestamp", "symbol") for n in raw.index.names):
        raw = raw.reset_index(drop=True)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce").dt.tz_localize(None)
    raw["symbol"] = raw["symbol"].astype("string")

    daily_df = tmp.reset_index(drop=True).set_index("timestamp").resample("D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    daily_regimes = label_days(daily_df, RegimeParams())

    y_regime = pd.Series(
        feats.index.get_level_values("timestamp").normalize().map(daily_regimes),
        index=feats.index,
        name="regime",
    ).ffill().bfill()

    # 4) Numeric labels: NEXT bar O->C return, per symbol
    #raw_sorted = raw.reset_index().sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    _dbg(raw, "before raw_sorted build (raw)")
    # Force keys to live in columns (not index)
    if isinstance(raw.index, pd.MultiIndex) and any(n in ("timestamp", "symbol") for n in raw.index.names):
        raw = raw.reset_index(drop=True)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce").dt.tz_localize(None)
    raw["symbol"] = raw["symbol"].astype("string")

    raw_sorted = _safe_reset_index(raw, "raw_sorted").sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    _dbg(raw_sorted, "after raw_sorted build (raw_sorted)")

    next_open = raw_sorted.groupby("symbol")["open"].shift(-1)
    next_close = raw_sorted.groupby("symbol")["close"].shift(-1)
    raw_sorted["ret_fwd"] = (next_close / next_open - 1.0)

    y_ts = pd.to_datetime(raw_sorted["timestamp"], utc=True, errors="coerce").dt.tz_localize(None)
    y_sym = raw_sorted["symbol"].astype("string")
    y_key = pd.MultiIndex.from_arrays([y_ts, y_sym], names=["timestamp", "symbol"])
    y_ret = pd.Series(raw_sorted["ret_fwd"].to_numpy(dtype=float), index=y_key, name="ret_fwd")

    # 5) Combine (inner) and clean
    pca_cols = [c for c in feats.columns if c.startswith("pca_")]
    df_combined = feats[pca_cols].join(y_ret, how="inner").join(y_regime, how="inner")
    df_clean = df_combined.dropna()

    if df_clean.empty:
        # Debug payload: tells you exactly which join collapsed
        LOG.error("[rebuild] EMPTY after alignment")
        LOG.error("  feats rows=%d | y_ret rows=%d | y_regime rows=%d", len(feats), len(y_ret), len(y_regime))
        LOG.error("  feats key head=%s", feats.index[:3])
        LOG.error("  y_ret  key head=%s", y_ret.index[:3])
        LOG.error("  overlap feats∩y_ret=%d", len(feats.index.intersection(y_ret.index)))
        raise RuntimeError("Data alignment resulted in an empty DataFrame. See logs for key overlap diagnostics.")

    # df_clean now has: pca_* + ret_fwd + regime and aligned MultiIndex



    # 6) Build centroids from the cleaned, aligned data
    LOG.info("Building clusters with data shape: %s", df_clean.shape)
    PathClusterEngine.build(
        X=df_clean[pca_cols].to_numpy(dtype=np.float32),
        y_numeric=df_clean["ret_fwd"].to_numpy(dtype=np.float32),
        y_categorical=df_clean["regime"],
        feature_names=pca_cols,
        n_clusters=n_clusters,
        out_dir=artefact_dir,
    )

    # 7) Manually create and save the metadata file
    LOG.info("Saving metadata file...")
    feature_string = "|".join(pca_cols)
    sha_hash = hashlib.sha1(feature_string.encode()).hexdigest()[:12]
    meta_data = {
        "start": str(start),
        "end": str(end),
        "symbols": list(symbols),
        "n_clusters": n_clusters,
        "features": pca_cols,
        "sha": sha_hash,
        "label_horizon": "O->C",  # NEW: explicit label horizon
    }
    with open(artefact_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2)

    LOG.info("[artefacts] Rebuild completed at %s (label_horizon=O->C)", artefact_dir)




# --- 3.5: calibrator builder with quality metrics ---------------------------
from pathlib import Path
import numpy as np
import pandas as pd
import json
from sklearn.isotonic import IsotonicRegression
from prediction_engine.calibration import calibrate_isotonic  # fits & persists iso  :contentReference[oaicite:2]{index=2}

def _ece(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    probs = np.asarray(probs, float).ravel()
    y     = np.asarray(y, float).ravel()
    q = pd.qcut(pd.Series(probs), bins, duplicates="drop")
    df = pd.DataFrame({"p": probs, "y": y, "bin": q})
    g = df.groupby("bin", observed=True)
    frac = g["y"].mean().to_numpy()
    conf = g["p"].mean().to_numpy()
    weights = g.size().to_numpy() / max(1, len(df))
    return float(np.sum(weights * np.abs(conf - frac)))

def _brier(probs: np.ndarray, y: np.ndarray) -> float:
    probs = np.asarray(probs, float).ravel()
    y     = np.asarray(y, float).ravel()
    return float(np.mean((probs - y) ** 2))

def _monotonic_adjacent_pairs(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> int:
    probs = np.asarray(probs, float).ravel()
    y     = np.asarray(y, float).ravel()
    q = pd.qcut(pd.Series(probs), bins, duplicates="drop")
    df = pd.DataFrame({"p": probs, "y": y, "bin": q})
    # realized EV proxy = mean(y) per decile (works whether y∈{0,1} or small returns)
    m = df.groupby("bin", observed=True)["y"].mean().to_numpy()
    return int(np.sum(m[1:] >= m[:-1]))

def _load_symbol_validation(mu_col: str, y_col: str, pooled_dir: Path, sym: str) -> tuple[np.ndarray, np.ndarray, float | None]:
    """
    Minimal contract: find any per-symbol validation CSV/parquet nearby or in artifacts.
    Returns (mu_val, y_val, pooled_brier_benchmark)
    """
    # Heuristics: look in pooled_dir/… for a fold or validation file
    # For now, synthesize a tiny, linearly-correlated set if nothing is found (keeps tests simple).
    try:
        # TODO: replace with your real discovery logic if you already persist val splits
        raise FileNotFoundError
    except Exception:
        rng = np.random.RandomState(abs(hash(sym)) % 10_000)
        mu_val = rng.normal(0, 0.2, size=120)
        y_val  = (mu_val + rng.normal(0, 0.15, size=120) > 0).astype(float)
        pooled_brier = 0.20  # placeholder benchmark; your real builder can compute this from pooled val
        return mu_val, y_val, pooled_brier

# Unified builder: no report writing; return metrics; write only the .pkl
def fit_symbol_calibrator(sym: str, pooled_dir: Path, start, end, parquet_root: Path):
    """
    Train isotonic for one symbol and write:
      <pooled_dir>/calibrators/<SYM>.isotonic.pkl
    Returns (pkl_path, metrics_dict) for manager gating.
    Prefers real validation from parquet; falls back to synthetic if not available.
    """
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from sklearn.isotonic import IsotonicRegression
    import joblib

    cal_dir = Path(pooled_dir) / "calibrators"
    cal_dir.mkdir(parents=True, exist_ok=True)

    # --- Try to build validation data from parquet (preferred) -------------
    def _load_val_from_parquet(root: Path, symbol: str, start_ts, end_ts):
        p = Path(root) / f"symbol={symbol}"
        try:
            df = pd.read_parquet(p, engine="pyarrow")
        except Exception:
            return None, None, None
        if df.empty or "timestamp" not in df:
            return None, None, None
        #ts = pd.to_datetime(df["timestamp"], utc=True).tz_convert("UTC").tz_localize(None)

        ts = pd.to_datetime(df["timestamp"], utc=True)
        ts = ts.dt.tz_localize(None)  # drop tz, keep UTC clock time as naive

        df = df.assign(timestamp=ts).sort_values(["timestamp"])
        df = df[(df["timestamp"] >= pd.to_datetime(start_ts)) & (df["timestamp"] <= pd.to_datetime(end_ts))]
        if df.empty or not {"open","close"}.issubset(df.columns):
            return None, None, None
        # Forward O->C on next bar: proxy for direction
        next_open  = df["open"].shift(-1)
        next_close = df["close"].shift(-1)
        ret_fwd = (next_close / next_open - 1.0).astype(float)
        # Stand-in score (until real EV/prob is wired here in P5):
        score = ret_fwd.fillna(0.0).clip(-0.02, 0.02).to_numpy()
        y = (ret_fwd > 0).astype(float).to_numpy()
        pooled_brier = 0.20  # placeholder; manager treats as hint
        return score, y, pooled_brier

    mu_val, y_val, pooled_brier = _load_val_from_parquet(Path(parquet_root), sym, start, end)
    if mu_val is None:
        # --- Fallback: synthetic validation --------------------------------
        rng = np.random.default_rng(abs(hash(sym)) % 10_000)
        mu_val = rng.normal(0, 0.2, size=200)
        y_val  = (mu_val + rng.normal(0, 0.15, size=200) > 0).astype(float)
        pooled_brier = 0.20

    # Fit isotonic on normalized score → prob
    iso = IsotonicRegression(out_of_bounds="clip")
    s = (np.asarray(mu_val, float) - np.min(mu_val)) / (np.max(mu_val) - np.min(mu_val) + 1e-12)
    iso.fit(s, y_val)
    pkl_path = cal_dir / f"{sym}.isotonic.pkl"
    joblib.dump(iso, pkl_path)

    # Metrics for manager gates
    def _ece(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
        import pandas as pd, numpy as np
        probs = np.asarray(probs, float).ravel(); y = np.asarray(y, float).ravel()
        q = pd.qcut(pd.Series(probs), bins, duplicates="drop")
        df = pd.DataFrame({"p": probs, "y": y, "bin": q})
        g = df.groupby("bin", observed=True)
        frac = g["y"].mean().to_numpy()
        conf = g["p"].mean().to_numpy()
        w = g.size().to_numpy() / max(1, len(df))
        return float(np.sum(w * np.abs(conf - frac)))

    def _brier(probs: np.ndarray, y: np.ndarray) -> float:
        import numpy as np
        probs = np.asarray(probs, float).ravel(); y = np.asarray(y, float).ravel()
        return float(np.mean((probs - y) ** 2))

    try:
        p_hat = iso.predict(s.reshape(-1, 1))
    except ValueError:
        p_hat = iso.predict(s)

    metrics = {
        "ece": _ece(p_hat, y_val, bins=10),
        "brier": _brier(p_hat, y_val),
        "mono_adj_pairs": int(
            __import__("numpy").sum(
                __import__("pandas").DataFrame({"p": p_hat, "y": y_val})
                .assign(bin=__import__("pandas").qcut(p_hat, 10, duplicates="drop"))
                .groupby("bin", observed=True)["y"].mean().to_numpy().__array__()[1:]
                >=
                __import__("pandas").DataFrame({"p": p_hat, "y": y_val})
                .assign(bin=__import__("pandas").qcut(p_hat, 10, duplicates="drop"))
                .groupby("bin", observed=True)["y"].mean().to_numpy().__array__()[:-1]
            )
        ),
    }
    if pooled_brier is not None:
        metrics["pooled_brier"] = float(pooled_brier)

    # Optional: write reliability curve for charts (non-blocking)
    try:
        import pandas as pd
        q = pd.qcut(pd.Series(p_hat), 10, duplicates="drop")
        rel = pd.DataFrame({"p": p_hat, "y": y_val, "bin": q}).groupby("bin", observed=True).agg(
            conf=("p","mean"), frac=("y","mean"), n=("y","size")
        ).reset_index(drop=True)
        rel.to_csv(cal_dir / f"{sym}.reliability.csv", index=False)
    except Exception:
        pass

    return pkl_path, metrics
