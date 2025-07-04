"""
Prediction Engine package (v0.3.0)
===================================

This release completes the *core math* and *operational plumbing* needed for a
minimum‑viable prediction loop.  New modules:
    • market_regime.py – intraday & EOD detectors + disk persistence
    • path_cluster_engine.py – builds centroids from historical parquet
    • ev_engine.py – converts probabilities → EV(mean,var) with slippage model
    • tx_cost.py – simple square‑root impact curve, nightly‑calibrated
    • weight_optimization.py – overnight recency/tail curve search + save_best()
    • drift_monitor.py – KL/KS drift tests with rolling reference window
    • retraining_manager.py – drift‑aware retrain, MLflow logging (opt‑in)
    • hyperparam_tuning.py – async grid/Bayes tuning via skopt
    • explainability.py – SHAP‑optional feature importance helpers

Existing modules (schemas, distance_calculator, index_backends, models) are
unchanged from v0.2.0.
"""
from __future__ import annotations

# ---------------------------
# FILE: prediction_engine/__init__.py
# ---------------------------
"""Prediction‑Engine package – v0.4.0
Sets up structured JSON logging and exposes a global CONFIG.
"""

import logging
import sys
from pathlib import Path

import structlog
import yaml

# ------------------------------------------------------------------
# 1. logging bootstrap (structlog JSON to stdout)
# ------------------------------------------------------------------
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
logger = structlog.get_logger("pred_engine")

# ------------------------------------------------------------------
# 2. config model
# ------------------------------------------------------------------
from .schemas import EngineConfig  # noqa: E402

_CFG_FILE = Path(__file__).with_suffix(".yaml")
CONFIG = EngineConfig(**yaml.safe_load(_CFG_FILE.read_text()) if _CFG_FILE.exists() else {})

__all__ = ["CONFIG", "EngineConfig", "logger"]


