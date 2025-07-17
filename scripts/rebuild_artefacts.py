# ---------------------------------------------------------------------------
# prediction_engine/scripts/rebuild_artefacts.py (Corrected)
# ---------------------------------------------------------------------------
from __future__ import annotations
import json, logging, hashlib # Import hashlib
from pathlib import Path
from typing import Sequence

import pandas as pd
import numpy as np

from prediction_engine.market_regime import label_days, RegimeParams
from feature_engineering.pipelines.core import CoreFeaturePipeline
from prediction_engine.path_cluster_engine import PathClusterEngine
from feature_engineering.labels.labeler import one_bar_ahead      # NEW
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
) -> None:
    """
    Rebuild PathClusterEngine artefacts if they are missing or outdated.
    """
    artefact_dir = Path(artefact_dir)
    artefact_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    meta = _load_meta(artefact_dir / "meta.json")

    # Only skip if meta.json exactly matches our request params
    if meta:
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
            return
        LOG.info("[artefacts] Artefacts outdated (range/symbols changed). Rebuilding…")


    LOG.warning("[artefacts] Artefacts missing — rebuilding artefacts …")

    # 1) Load raw minute-bar parquet slice
    raw_pq_path = Path(parquet_root)
    filt = [("symbol", "in", list(symbols)),
            ("timestamp", ">=", start),
            ("timestamp", "<=", end)]
    raw = pd.read_parquet(raw_pq_path, filters=filt)
    if raw.empty:
        raise RuntimeError("Rebuild slice returned zero rows – check dates/symbols.")

    # Ensure timestamp is a datetime object
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])

    # Add the missing 'trigger_ts' and 'volume_spike_pct' columns
    raw["trigger_ts"] = raw["timestamp"]
    volume_ma = raw['volume'].rolling(window=20, min_periods=1).mean()
    raw['volume_spike_pct'] = (raw['volume'] / volume_ma) - 1.0
    raw['volume_spike_pct'] = raw['volume_spike_pct'].fillna(0.0)


    # 2) Run in-memory pipeline to get PCA features
    pipe = CoreFeaturePipeline(parquet_root=Path(""))
    feats, _ = pipe.run_mem(raw)

    # 3) Resample minute data to DAILY to generate correct regime labels
    LOG.info("Resampling to daily for regime calculation...")
    daily_df = raw.set_index('timestamp').resample('D').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna()

    daily_regimes = label_days(daily_df, RegimeParams())

    # Map daily regimes back to each minute bar
    feats_indexed = feats.set_index('timestamp')
    normalized_dates = feats_indexed.index.normalize()
    regime_series = pd.Series(
        normalized_dates.map(daily_regimes).values,
        index = feats_indexed.index,
        name = 'regime'
    )
    y_categorical = regime_series.ffill().bfill()

    # 4) Identify the PCA columns from the result
    pca_cols = [c for c in feats.columns if c.startswith("pca_")]
    if not pca_cols:
        raise RuntimeError("CoreFeaturePipeline did not produce any pca_* columns.")

    # 5) Prepare final aligned data for clustering
    LOG.info("Preparing and aligning data for clustering...")

    # --- FIX: Synchronize timezones before mapping and joining ---
    LOG.info("Normalizing timezones for data alignment...")

    # Ensure both the feature DataFrame and regime Series use timezone-naive indexes
    if 'timestamp' in feats.columns:
        feats = feats.set_index('timestamp')
    if feats.index.tz is not None:
        feats.index = feats.index.tz_localize(None)
    if daily_regimes.index.tz is not None:
        daily_regimes.index = daily_regimes.index.tz_localize(None)

    # Now that timezones are aligned, create the categorical labels
    y_categorical = pd.Series(
        feats.index.normalize().map(daily_regimes),  # Map daily regimes to minute-bar dates
        index=feats.index,
        name='regime'
    ).ffill().bfill()  # Fill any gaps

    # --- FIX: Generate numeric labels with the correct DatetimeIndex ---
    y_numeric_series = pd.Series(
        one_bar_ahead(raw).values,
        index=pd.to_datetime(raw['timestamp']),  # Use the timestamp column for the index
        name="ret_fwd"
    )

    # Also normalize this index to ensure it matches `feats`
    if y_numeric_series.index.tz is not None:
        y_numeric_series.index = y_numeric_series.index.tz_localize(None)

    # Now all three components (feats, y_numeric_series, y_categorical) have a
    # matching, timezone-naive DatetimeIndex, and the join will work.
    df_combined = feats.join(y_numeric_series).join(y_categorical)

    # Drop any row that has a NaN in the required features or labels
    required_cols = pca_cols + ["ret_fwd", "regime"]
    df_clean = df_combined[required_cols].dropna()

    # 6) Build centroids from the cleaned, aligned data
    LOG.info("Building clusters with data shape: %s", df_clean.shape)

    # Add a check to provide a better error if the dataframe is still empty
    if df_clean.empty:
        raise RuntimeError(
            "Data alignment resulted in an empty DataFrame. "
            "Check for issues with NaNs or index mismatches in the source data."
        )

    print("DIAGNOSTIC ret_fwd\n", df_clean["ret_fwd"].describe())
    print("DIAGNOSTIC A:\n", df_clean["ret_fwd"].head(), df_clean["ret_fwd"].tail())

    PathClusterEngine.build(
        # --- FIX: Create X from the 'df_clean' DataFrame, not 'feats' ---
        X=df_clean[pca_cols].to_numpy(dtype=np.float32),

        y_numeric=df_clean["ret_fwd"].to_numpy(dtype=np.float32),
        y_categorical=df_clean["regime"],
        feature_names=pca_cols,
        n_clusters=n_clusters,
        out_dir=artefact_dir,
    )

    '''PathClusterEngine.build(
        X=feats[pca_cols].to_numpy(),
        y_numeric=feats[pca_cols[0]].to_numpy(),
        y_categorical=y_categorical,
        feature_names=pca_cols,
        n_clusters=n_clusters,
        out_dir=artefact_dir,
    )'''

    # 6) Manually create and save the metadata file after building artefacts
    LOG.info("Saving metadata file...")

    # --- FIX: Use the exact same hashing logic as ev_engine.py ---
    feature_string = "|".join(pca_cols)
    sha_hash = hashlib.sha1(feature_string.encode()).hexdigest()[:12]

    meta_data = {
        "start": str(start),
        "end": str(end),
        "symbols": list(symbols),
        "n_clusters": n_clusters,
        "features": pca_cols,
        "sha": sha_hash, # Add the correctly calculated SHA hash
    }
    with open(artefact_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2)

    LOG.info("[artefacts] Rebuild completed at %s", artefact_dir)
