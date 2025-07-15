# ---------------------------------------------------------------------------
# prediction_engine/scripts/rebuild_artefacts.py (Corrected)
# ---------------------------------------------------------------------------
from __future__ import annotations
import json, logging
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
    meta = _load_meta(artefact_dir / "meta.json")

    #if meta:
    #    LOG.info("[artefacts] Artefacts exist. Skipping rebuild.")
    #    return
    # Only skip if meta.json exactly matches our request params
    if meta:
        if (
            meta.get("start") == str(start)
            and meta.get("end") == str(end)
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

    # 2) Run in-memory pipeline to get PCA features
    pipe = CoreFeaturePipeline(parquet_root=Path(""))
    feats, _ = pipe.run_mem(raw)

    # 3) Resample minute data to DAILY to generate correct regime labels
    LOG.info("Resampling to daily for regime calculation...")
    daily_df = raw.set_index('timestamp').resample('D').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
    }).dropna()

    daily_regimes = label_days(daily_df, RegimeParams())

    # Map daily regimes back to each minute bar in the features dataframe
    #feats_indexed = feats.set_index('timestamp')
    #minute_regimes = feats_indexed.index.normalize().map(daily_regimes)

    #y_categorical = minute_regimes.ffill().bfill()

    # Map daily regimes back to each minute bar in the features dataframe
    feats_indexed = feats.set_index('timestamp')
    normalized_dates = feats_indexed.index.normalize()

    # Build a Series indexed by the minute timestamps
    regime_series = pd.Series(
        normalized_dates.map(daily_regimes).values,
        index = feats_indexed.index,
        name = 'regime'
    )
    # Forward/backward fill to eliminate any NaNs
    y_categorical = regime_series.ffill().bfill()

    # 4) Identify the PCA columns from the result
    pca_cols = [c for c in feats.columns if c.startswith("pca_")]
    if not pca_cols:
        raise RuntimeError("CoreFeaturePipeline did not produce any pca_* columns.")

    # 5) Build centroids using the PCA features and correct regime labels
    LOG.info("Building clusters with regime labels...")
    PathClusterEngine.build(
        X=feats[pca_cols].to_numpy(),
        y_numeric=feats[pca_cols[0]].to_numpy(),  # Using a placeholder y_numeric
        y_categorical=y_categorical,
        feature_names=pca_cols,
        n_clusters=n_clusters,
        out_dir=artefact_dir,
    )

    LOG.info("[artefacts] Rebuild completed at %s", artefact_dir)
