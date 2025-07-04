# ---------------------------------------------------------------------------
# retraining_manager.py
# ---------------------------------------------------------------------------
"""Handle model retrain triggers + optional MLflow logging."""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import mlflow  # noqa: F401 – optional, ignore if env not set

from .models import ModelManager


class RetrainingManager:
    def __init__(self, mm: ModelManager, drift_thresh: float):
        self.mm = mm
        self.drift_thresh = drift_thresh

    # ----------------------------------------------
    def check_and_retrain(self, now: datetime, perf: dict[str, Any], drift_val: float, regime_changed: bool, hist_df):
        if drift_val > self.drift_thresh or regime_changed:
            X, y = hist_df.iloc[:, :-1].to_numpy(), hist_df.iloc[:, -1].to_numpy()
            self.mm.train_model(X, y)
            if os.getenv("MLFLOW_TRACKING_URI"):
                with mlflow.start_run(run_name=f"retrain_{now:%Y%m%d_%H%M%S}"):
                    mlflow.log_metric("drift", drift_val)
                    mlflow.log_metrics(perf)
                    mlflow.log_params({"regime_change": regime_changed})

