# after other imports in the tests package
import json
from pathlib import Path
import numpy as np

from prediction_engine.ev_engine import _coerce_curve_params_dict
from prediction_engine.weight_optimization import CurveParams

WEIGHTS_ROOT = Path("weights")

def _load_curve_params_for(regime: str) -> CurveParams:
    p = WEIGHTS_ROOT / f"regime={regime}" / "curve_params.json"
    raw = {}
    if p.exists():
        raw = json.loads(p.read_text())
    params_dict = _coerce_curve_params_dict(raw)
    print("Loading from", p.resolve())

    return CurveParams(**params_dict)

def test_regime_curves_are_not_identical():
    """Ensure TREND / RANGE / VOL are configured with distinct curve shapes."""
    trend = _load_curve_params_for("TREND")
    range_ = _load_curve_params_for("RANGE")
    vol   = _load_curve_params_for("VOL")

    triples = {
        "TREND": (trend.family, trend.tail_len, trend.shape, trend.blend_alpha),
        "RANGE": (range_.family, range_.tail_len, range_.shape, range_.blend_alpha),
        "VOL": (vol.family, vol.tail_len, vol.shape, vol.blend_alpha),
    }

    unique = {triples[k] for k in triples}
    assert len(unique) >= 2, f"Regime curves appear identical: {triples}"

