import numpy as np
from prediction_engine.distance_calculator import DistanceCalculator

def _spearman_ranks(a, b):
    # simple rank corr (no ties handling sophistication needed for this toy)
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    num = float((ra*rb).sum())
    den = float(np.sqrt((ra*ra).sum() * (rb*rb).sum()))
    return num/den if den > 0 else 0.0

def test_rf_weighted_distances_track_importance():
    rng = np.random.default_rng(0)
    # Feature 0 is 9x more important than feature 1
    w = np.array([0.9, 0.1], dtype=np.float32)
    # Build centers varying mostly along feature 0
    n = 200
    centers = np.c_[rng.normal(0, 2.0, size=n), rng.normal(0, 0.2, size=n)].astype(np.float32)
    x = np.array([0.0, 0.0], dtype=np.float32)

    # Dist calc with RF weights
    dc = DistanceCalculator(centers, metric="rf_weighted", rf_weights=w)

    '''d2, idx = dc.batch_top_k(x[np.newaxis,:], k=n)
    d2 = d2.ravel()

    # Compare distances to |feature0| magnitude (proxy for importance)
    proxy = np.abs(centers[:,0])'''

    d2, idx = dc.batch_top_k(x[np.newaxis, :], k=n)
    d2 = d2.ravel()
    idx = idx.ravel()
    # Compare distances to |feature0| magnitude (aligned to returned neighbors)
    proxy = np.abs(centers[idx, 0])

    rho = _spearman_ranks(d2, proxy)
    assert rho > 0.4, f"Spearman too low: {rho:.3f}"
