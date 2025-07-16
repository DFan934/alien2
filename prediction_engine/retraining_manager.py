# ---------------------------------------------------------------------------
# prediction_engine/retraining_manager.py
# ---------------------------------------------------------------------------
"""Handle model-retrain triggers + atomic artefact hot-swap."""
from __future__ import annotations

import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import mlflow  # noqa: F401 – only active if env-vars are set
import numpy as np
import pandas as pd
from prediction_engine.weight_optimization import WeightOptimizer as wo

from .models import ModelManager
from .path_cluster_engine import PathClusterEngine


class RetrainingManager:
    """
    Watches drift / regime flags and, when tripped, rebuilds the path-cluster
    artefacts in a **temporary directory** before atomically swapping them into
    production.  If MLflow tracking is configured the run is logged.
    """

    def __init__(self, mm: ModelManager, drift_thresh: float):
        self.mm = mm
        self.drift_thresh = float(drift_thresh)
        # ─── load recency curves on startup ────────────────────────
        # assume weights were written to mm’s artefact directory
        artefact_root = Path(mm.artefact_dir)  # if you track that path
        self.recency_curves = wo.load_weight_curves(artefact_root / "weights")

    def reload_curves(self):
        """Re-load the JSON weight curves (e.g. after nightly run finishes)."""
        artefact_root = Path(self.mm.artefact_dir) / "weights"
        self.recency_curves = wo.load_weight_curves(artefact_root)


    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #
    def retrain(
        self,
        *,
        X: np.ndarray,
        y_numeric: np.ndarray,
        y_categorical: Optional[pd.Series],
        feature_names: list[str],
        n_clusters: int,
        artefact_dir: Path,
        ml_tags: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Build new centroids + stats in a temp dir and hot-swap them in one move.

        Parameters
        ----------
        artefact_dir
            Directory containing the *live* artefacts produced by
            ``PathClusterEngine.build``.  The function creates
            ``tmp_retrain_*`` beside it, writes new files there, then uses
            :pyfunc:`Path.rename` (atomic on POSIX) to replace the old folder.
        """
        artefact_dir = artefact_dir.expanduser().resolve()
        parent = artefact_dir.parent

        # --- 1. Train in isolated temp folder ------------------------
        with tempfile.TemporaryDirectory(dir=parent, prefix="tmp_retrain_") as tmp:
            tmp_path = Path(tmp)

            PathClusterEngine.build(
                X=X,
                y_numeric=y_numeric,
                y_categorical=y_categorical,
                feature_names=feature_names,
                n_clusters=n_clusters,
                out_dir=tmp_path,
            )

            # --- 2. Optional: MLflow logging ------------------------
            if os.getenv("MLFLOW_TRACKING_URI"):
                with mlflow.start_run(run_name="centroid_retrain"):
                    mlflow.log_params(
                        {
                            "n_clusters": n_clusters,
                            "n_rows": X.shape[0],
                            "n_features": X.shape[1],
                        }
                    )
                    if ml_tags:
                        mlflow.set_tags(ml_tags)
                    mlflow.log_artifacts(tmp_path, artifact_path="centroid_artifacts")

            # --- 3. Atomic hot-swap ---------------------------------
            backup = artefact_dir.with_suffix(".bak")
            try:
                if artefact_dir.exists():
                    artefact_dir.replace(backup)  # rename → cheap & atomic
                tmp_path.replace(artefact_dir)     # promote temp to live
            finally:
                # best-effort cleanup – ignore errors so swap never blocks
                if backup.exists():
                    shutil.rmtree(backup, ignore_errors=True)

    # ------------------------------------------------------------------ #
    # Convenience check – call from DriftMonitor or regime switch        #
    # ------------------------------------------------------------------ #
    def check_and_retrain(
        self,
        now: datetime,
        perf: dict[str, Any],
        drift_val: float,
        regime_changed: bool,
        hist_df: pd.DataFrame,
        artefact_dir: Path,
    ) -> None:
        """
        Trigger retrain when *drift* exceeds threshold **or** regime flip occurs.
        """
        trigger = (drift_val > self.drift_thresh) or regime_changed
        if not trigger:
            return

        X = hist_df.iloc[:, :-1].to_numpy(dtype=np.float32)
        y_num = hist_df.iloc[:, -1].to_numpy(dtype=np.float32)

        # optional categorical regime/outcome col
        y_cat = (
            hist_df["regime"] if "regime" in hist_df.columns else None
        )

        feature_names = hist_df.columns[:-1].tolist()
        n_clusters = int(np.sqrt(len(hist_df))) or 8

        self.retrain(
            X=X,
            y_numeric=y_num,
            y_categorical=y_cat,
            feature_names=feature_names,
            n_clusters=n_clusters,
            artefact_dir=artefact_dir,
            ml_tags={
                "drift": f"{drift_val:.4f}",
                "regime_change": str(regime_changed),
                "timestamp": now.isoformat(timespec="seconds"),
            },
        )

        # also refresh any in-memory models managed by ModelManager
        self.mm.load_latest(artefact_dir)
