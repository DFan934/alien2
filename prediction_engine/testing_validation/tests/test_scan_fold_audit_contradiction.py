import pandas as pd
import pytest

# Import from your walkforward module where you added it
from prediction_engine.testing_validation.walkforward import _audit_scan_fold_intersection
#from s import _audit_scan_fold_intersection


def test_scan_fold_contradiction_hard_fails():
    # FULL is tz-aware UTC, SCAN is naive → after coercion both are UTC,
    # but we deliberately make them non-overlapping so intersection is empty.
    full = pd.DataFrame({"timestamp": pd.to_datetime(["2020-01-01 10:00:00+00:00", "2020-01-01 10:01:00+00:00"])})
    scan = pd.DataFrame({"timestamp": pd.to_datetime(["2020-01-02 10:00:00", "2020-01-02 10:01:00"])})  # naive

    full_tr = full.copy()
    full_te = full.copy()
    scan_tr = scan.copy()
    scan_te = scan.copy()

    with pytest.raises(RuntimeError, match="SCAN↔FOLD CONTRADICTION"):
        _audit_scan_fold_intersection(
            full_df=full,
            scan_df=scan,
            full_tr=full_tr,
            full_te=full_te,
            scan_tr=scan_tr,
            scan_te=scan_te,
            allow_fallback=False,
            fold_idx=1,
        )


def test_scan_fold_contradiction_can_be_overridden_in_dev_mode():
    full = pd.DataFrame({"timestamp": pd.to_datetime(["2020-01-01 10:00:00+00:00"])})
    scan = pd.DataFrame({"timestamp": pd.to_datetime(["2020-01-02 10:00:00"])})  # no overlap

    scan_tr_in, scan_te_in = _audit_scan_fold_intersection(
        full_df=full,
        scan_df=scan,
        full_tr=full,
        full_te=full,
        scan_tr=scan,
        scan_te=scan,
        allow_fallback=True,   # dev escape hatch
        fold_idx=1,
    )
    assert scan_tr_in == []
    assert scan_te_in == []
