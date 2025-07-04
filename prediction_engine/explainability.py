# ---------------------------------------------------------------------------
# explainability.py
# ---------------------------------------------------------------------------
"""SHAP‑optional explainability helpers."""
from __future__ import annotations

from typing import List

import numpy as np

try:
    import shap  # type: ignore
except ImportError:  # pragma: no cover
    shap = None


def get_top_influencers(model, X_sample: np.ndarray, n: int = 5) -> List[int]:
    if shap is None:
        # fallback: use absolute coef_/feature_importances_
        if hasattr(model, "coef_"):
            imp = np.abs(model.coef_).ravel()
        elif hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        else:
            return []
    else:
        explainer = shap.Explainer(model, X_sample)
        shap_vals = explainer(X_sample)
        imp = np.abs(shap_vals.values).mean(axis=0)
    return np.argsort(imp)[::-1][:n].tolist()
