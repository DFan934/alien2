import numpy as np
from prediction_engine.ev_engine import EVEngine
# from prediction_engine.market_regime import MarketRegime  # not needed

def _engine_with_all_regimes():
    centers = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]], dtype=np.float32)
    mu = np.array([0.01, -0.01, 0.02], dtype=np.float32)
    var = np.array([0.05, 0.05, 0.05], dtype=np.float32)
    var_down = np.array([0.04, 0.04, 0.04], dtype=np.float32)
    regimes = np.array(["TREND","RANGE","VOL"])
    scales = np.ones(2, dtype=np.float32)
    return EVEngine(
        centers=centers, mu=mu, var=var, var_down=var_down,
        h=1.0, metric="euclidean", k=3,
        _n_feat_schema=2, _scale_vec=scales, cluster_regime=regimes
    )

def test_train_fallback_rate_threshold():
    eng = _engine_with_all_regimes()
    xs = []
    xs += [np.array([0.05, 0.05], dtype=np.float32)] * 50   # TREND vicinity
    xs += [np.array([5.1, 0.05], dtype=np.float32)] * 50    # RANGE vicinity
    xs += [np.array([0.05, 5.1], dtype=np.float32)] * 50    # VOL vicinity

    # Evaluate under the matching requested regime (use strings)
    for i, x in enumerate(xs):
        if i < 50:
            eng.evaluate(x, regime="TREND")
        elif i < 100:
            eng.evaluate(x, regime="RANGE")
        else:
            eng.evaluate(x, regime="VOL")

    total = len(eng._search_log)
    fallbacks = sum(1 for r in eng._search_log if r.get("fallback_depth", 0) > 0)
    fallback_rate = fallbacks / max(1, total)
    assert fallback_rate <= 0.10, f"fallback_rate={fallback_rate:.3f} exceeded 10%"
