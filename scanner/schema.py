# =============================================================
# NEW FILE: scanner/schema.py
# -------------------------------------------------------------
# Single source‑of‑truth for the feature order expected by both
# the scanner snapshots *and* the downstream CoreFeaturePipeline /
# EVEngine.  Import this constant everywhere you need to guarantee
# column alignment.
# =============================================================
from __future__ import annotations

from feature_engineering.pipelines.core import _PREDICT_COLS

# ------------------------------------------------------------------
# FEATURE_ORDER  –  immutable, canonical column list
# ------------------------------------------------------------------
#   • First  : engineered feature columns used by EVEngine
#   • Last   : administrative columns needed throughout the stack
# ------------------------------------------------------------------
FEATURE_ORDER: list[str] = list(_PREDICT_COLS) + ["symbol", "timestamp"]

__all__ = ["FEATURE_ORDER"]

# Guard against accidental mutation
FEATURE_ORDER = tuple(FEATURE_ORDER)  # type: ignore[assignment]