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

    # 5) Build centroids using the PCA features and correct regime labels
    LOG.info("Building clusters with regime labels...")
    PathClusterEngine.build(
        X=feats[pca_cols].to_numpy(),
        y_numeric=feats[pca_cols[0]].to_numpy(),
        y_categorical=y_categorical,
        feature_names=pca_cols,
        n_clusters=n_clusters,
        out_dir=artefact_dir,
    )

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
