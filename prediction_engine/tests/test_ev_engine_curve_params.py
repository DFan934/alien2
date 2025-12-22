# FILE: prediction_engine/tests/test_ev_engine_curve_params.py

from prediction_engine.ev_engine import _coerce_curve_params_dict, CurveParams  # CurveParams re-exported via ev_engine


def test_coerce_curve_params_flat_payload():
    """Flat, legacy-style payload should coerce into a valid CurveParams."""
    raw = {
        "family": "exp",
        "tail_len_days": 20,
        "alpha": 1.7,  # legacy field; should map to blend_alpha
    }
    kwargs = _coerce_curve_params_dict(raw)
    cp = CurveParams(**kwargs)

    assert cp.family == "exp"
    assert cp.tail_len == 20
    # No shape in raw -> default
    assert cp.shape == 1.0
    # Alpha should have become blend_alpha
    assert cp.blend_alpha == 1.7
    # Lambda default
    assert cp.lambda_reg == 1.0


def test_coerce_curve_params_nested_params_block():
    """New-style payload with params block and explicit shape should pass through untouched."""
    raw = {
        "params": {
            "family": "exp",
            "tail_len": 30,
            "shape": 1.8,
            "blend_alpha": 0.4,
            "lambda_reg": 0.5,
        }
    }
    kwargs = _coerce_curve_params_dict(raw)
    cp = CurveParams(**kwargs)

    assert cp.family == "exp"
    assert cp.tail_len == 30
    assert cp.shape == 1.8
    assert cp.blend_alpha == 0.4
    assert cp.lambda_reg == 0.5
