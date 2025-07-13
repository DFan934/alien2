# ---------------------------------------------------------------------------
# prediction_engine/artifacts_sync.py
# ---------------------------------------------------------------------------
"""
Helpers that guarantee EVEngine artefacts are in sync with the live feature
schema.  If drift is detected, artefacts are rebuilt automatically.

This avoids the need to call a separate CLI tool.
"""
from __future__ import annotations
import json, logging
from pathlib import Path
from datetime import datetime
from typing import Sequence

import pandas as pd
import numpy as np

from scanner.schema import FEATURE_ORDER
from feature_engineering.pipelines.core import CoreFeaturePipeline
from prediction_engine.path_cluster_engine import PathClusterEngine

LOG = logging.getLogger(__name__)


def _load_meta(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def _sha1_list(names: Sequence[str]) -> str:
    import hashlib, itertools
    return hashlib.sha1("|".join(names).encode()).hexdigest()[:12]


def rebuild_if_needed(
    artefact_dir: str | Path,
    parquet_root: str | Path,
    symbols: Sequence[str],
    start: str,
    end: str,
    n_clusters: int = 64,
) -> None:
    """
    Rebuild PathClusterEngine artefacts *iff* feature list drifted.

    Parameters
    ----------
    artefact_dir
        Directory where centres.npy, cluster_stats.npz, meta.json live.
    parquet_root
        Root of raw minute-bar parquet dataset.
    symbols
        Universe to use for retrain slice.
    start, end
        Date range (YYYY-MM-DD strings) for the slice.
    """
    artefact_dir = Path(artefact_dir)
    meta = _load_meta(artefact_dir / "meta.json")
    live_feat = list(FEATURE_ORDER)

    if meta and meta.get("features") == live_feat:
        LOG.info("[artefacts] Feature list unchanged – no rebuild needed.")
        return  # in sync ✅

    LOG.warning("[artefacts] Feature drift detected — rebuilding artefacts …")

    # 1) load raw parquet slice
    filt = [("symbol", "in", list(symbols)),
            ("timestamp", ">=", start),
            ("timestamp", "<=", end)]
    raw = pd.read_parquet(parquet_root, filters=filt)
    if raw.empty:
        raise RuntimeError("Rebuild slice returned zero rows – check dates/symbols.")

    # 2) run in-memory pipeline
    pipe = CoreFeaturePipeline(parquet_root=Path("."))  # dummy, in-mem
    feats, _ = pipe.run_mem(raw)
    feats = feats[live_feat].astype(np.float32)

    # 3) rebuild centroids + stats
    PathClusterEngine.build(
        X=feats[live_feat].to_numpy(),
        y_numeric=feats[live_feat[0]].to_numpy(),   # simple target; adapt as needed
        y_categorical=None,
        feature_names=live_feat,
        n_clusters=n_clusters,
        out_dir=artefact_dir,
    )
    LOG.info("[artefacts] Rebuild completed at %s", artefact_dir)
