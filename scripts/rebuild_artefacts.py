# ---------------------------------------------------------------------------
# prediction_engine/scripts/rebuild_artefacts.py (Corrected)
# ---------------------------------------------------------------------------
from __future__ import annotations
import json
import logging
import hashlib
from pathlib import Path
from typing import Sequence

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
    if fitted_pipeline_dir is not None:
        pipe = CoreFeaturePipeline(parquet_root=Path(fitted_pipeline_dir))
        feats = pipe.transform_mem(raw)  # ← NO FIT
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
    '''raw = raw.sort_values(["symbol", "timestamp"])
    raw["ret_fwd"] = raw.groupby("symbol")["open"].shift(-1) / raw["open"] - 1.0
    y_numeric_series = pd.Series(
        raw["ret_fwd"].values,
        index=pd.to_datetime(raw["timestamp"]).dt.tz_localize(None),
        name="ret_fwd"
    )'''

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
        raise RuntimeError("Data alignment resulted in an empty DataFrame. Check for NaNs or index mismatches.")

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
