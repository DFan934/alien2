# -----------------------------------------------------------------------------
# NEW TEST: scanner/tests/test_feature_contract.py
# -----------------------------------------------------------------------------
# Verifies that scanner.schema.FEATURE_ORDER is identical to the
# _PREDICT_COLS list used by CoreFeaturePipeline.
# This catches silent drift in feature order or naming.
# -----------------------------------------------------------------------------

"""Contract test: Scanner ⇄ Feature‑Pipeline column order."""
import pytest

from scanner.schema import FEATURE_ORDER
from feature_engineering.pipelines.core import _PREDICT_COLS


def test_feature_contract():
    assert list(FEATURE_ORDER[:-2]) == list(_PREDICT_COLS), (
        "FEATURE_ORDER does not match CoreFeaturePipeline._PREDICT_COLS"
    )
    assert FEATURE_ORDER[-2:] == ("symbol", "timestamp"), "Admin columns missing or mis‑ordered"