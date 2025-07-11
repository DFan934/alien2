import numpy as np
from prediction_engine.ev_engine import EVEngine, EVResult
from prediction_engine.market_regime import MarketRegime

def _dummy_engine():
    centers = np.array([[0,0],[1,1],[2,2]], dtype=np.float32)
    mu = np.array([0.01, 0.02, 0.03], dtype=np.float32)
    var = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    regime = np.array(["TREND", "RANGE", "TREND"], dtype="U10")
    return EVEngine(centers=centers, mu=mu, var=var, var_down=var,
                    h=1.0, cluster_regime=regime)

def test_regime_mask():
    eng = _dummy_engine()
    res_trend: EVResult = eng.evaluate(np.array([0.2,0.2], dtype=np.float32), regime=MarketRegime.TREND)
    res_range: EVResult = eng.evaluate(np.array([0.2,0.2], dtype=np.float32), regime=MarketRegime.RANGE)
    # Different nearest-centroid IDs prove mask worked
    assert res_trend.cluster_id != res_range.cluster_id


# --- replace the entire second test ---------------------------------

def test_regime_mask_keeps_only_matching_clusters():
    eng = _dummy_engine()           # ← use helper directly (no fixture)
    x   = eng.centers[0]            # query near cluster 0
    res = eng.evaluate(x, regime=MarketRegime.TREND)

    # Ensure the selected cluster is labelled “TREND”
    assert eng.cluster_regime[res.cluster_id] == "TREND"
