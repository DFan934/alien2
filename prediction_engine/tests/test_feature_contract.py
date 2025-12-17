from feature_engineering.pipelines.core import FEATURE_ORDER as FE_FEATURE_ORDER
from scanner.schema import FEATURE_ORDER as SCANNER_FEATURE_ORDER

def test_scanner_and_fe_feature_order_match():
    assert list(SCANNER_FEATURE_ORDER) == list(FE_FEATURE_ORDER)

def test_feature_order_starts_with_timestamp_symbol():
    assert FE_FEATURE_ORDER[0] == "timestamp"
    assert FE_FEATURE_ORDER[1] == "symbol"


from scanner.schema import FEATURE_ORDER
from feature_engineering.pipelines.core import _PREDICT_COLS

def test_feature_order_contains_predict_cols_in_order():
    # must be a pure list, no dups
    assert len(FEATURE_ORDER) == len(set(FEATURE_ORDER)), "Duplicate columns in FEATURE_ORDER"

    # FE predict columns must all exist in FEATURE_ORDER
    missing = [c for c in _PREDICT_COLS if c not in FEATURE_ORDER]
    assert not missing, f"_PREDICT_COLS missing from FEATURE_ORDER: {missing}"

    # and preserve order (EV + downstream assumes stable ordering)
    idxs = [FEATURE_ORDER.index(c) for c in _PREDICT_COLS]
    assert idxs == sorted(idxs), "FEATURE_ORDER does not preserve _PREDICT_COLS order"

    # must be a pure list, no dups
    assert len(FEATURE_ORDER) == len(set(FEATURE_ORDER)), "Duplicate columns in FEATURE_ORDER"

    # FE predict columns must all exist in FEATURE_ORDER
    missing = [c for c in _PREDICT_COLS if c not in FEATURE_ORDER]
    assert not missing, f"_PREDICT_COLS missing from FEATURE_ORDER: {missing}"

    # and preserve order (EV + downstream assumes stable ordering)
    idxs = [FEATURE_ORDER.index(c) for c in _PREDICT_COLS]
    assert idxs == sorted(idxs), "FEATURE_ORDER does not preserve _PREDICT_COLS order"



