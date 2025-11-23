# prediction_engine/tests/test_walkforward_embargo.py

import pandas as pd
from pathlib import Path

from prediction_engine.testing_validation.walkforward import make_time_folds, save_folds_json, Fold

def _as_utc(ts):
    # Ensure naive UTC-like comparison (your code stores naive Timestamps)
    return pd.to_datetime(ts)

def test_make_time_folds_embargo_and_nonoverlap(tmp_path: Path):
    ws = pd.Timestamp("2000-01-03")  # Monday
    we = pd.Timestamp("2000-05-01")

    folds = make_time_folds(
        window_start=ws,
        window_end=we,
        mode="expanding",
        n_folds=3,
        test_span_days=10,
        train_min_days=30,
        embargo_days=3,
    )
    # Persist (acceptance)
    out = save_folds_json(tmp_path, folds)
    assert out.exists(), "folds.json was not written"

    # Basic sanity: non-empty, ordered
    assert len(folds) >= 1
    for f in folds:
        assert isinstance(f, Fold)
        assert f.train_start < f.train_end
        assert f.test_start < f.test_end
        # TRAIN and TEST strictly separated by embargo_days
        # i.e., last TRAIN day <= test_start - embargo_days
        embargo_gap = pd.Timedelta(days=int(f.purge_bars))
        assert _as_utc(f.train_end) <= _as_utc(f.test_start) - embargo_gap

    # TEST windows must not overlap
    for i in range(1, len(folds)):
        prev = folds[i-1]
        cur = folds[i]
        assert _as_utc(prev.test_end) < _as_utc(cur.test_start), "Test windows must be non-overlapping"

def test_make_time_folds_rolling_respects_train_length():
    ws = pd.Timestamp("2000-01-03")
    we = pd.Timestamp("2000-03-15")

    # short window, ensure rolling enforces train_min_days
    folds = make_time_folds(
        window_start=ws,
        window_end=we,
        mode="rolling",
        n_folds=2,
        test_span_days=10,
        train_min_days=20,
        embargo_days=2,
    )
    assert len(folds) >= 1
    for f in folds:
        # rolling: TRAIN length approximately train_min_days (±1 day due to right-closed boundary)
        approx_days = (f.train_end.normalize() - f.train_start.normalize()).days + 1
        assert approx_days >= 19, f"train window too small: {approx_days}d"
