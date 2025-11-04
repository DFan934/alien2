# prediction_engine/tests/test_ev_engine_regime_loading.py
import numpy as np
from prediction_engine.ev_engine import EVEngine
from prediction_engine.market_regime import MarketRegime

def _toy_engine(regimes):
    # centers & stats (tiny, deterministic)
    centers = np.array([[0.0,0.0],
                        [1.0,0.0],
                        [0.0,1.0]], dtype=np.float32)
    mu = np.array([+0.01, -0.02, +0.03], dtype=np.float32)
    var = np.array([0.04, 0.05, 0.06], dtype=np.float32)
    var_down = np.array([0.03,0.04,0.05], dtype=np.float32)
    cluster_regime = np.array(regimes, dtype="U10")
    scales = np.ones(centers.shape[1], dtype=np.float32)
    return EVEngine(
        centers=centers,
        mu=mu, var=var, var_down=var_down,
        h=1.0, metric="euclidean", k=2,
        _n_feat_schema=centers.shape[1],
        _scale_vec=scales,
        cluster_regime=cluster_regime,
    )

def test_regime_index_loading_and_fallback_counters():
    # No TREND clusters present → TREND request must fallback; RANGE available
    eng = _toy_engine(regimes=["RANGE", "RANGE", "VOL"])
    x = np.array([0.1, 0.1], dtype=np.float32)

    # Request TREND 5x to accumulate fallback counters
    for _ in range(5):
        eng.evaluate(x, regime=MarketRegime.TREND)
    # Request RANGE 5x → direct hit, no fallback
    for _ in range(5):
        eng.evaluate(x, regime=MarketRegime.RANGE)

    # Fallbacks recorded
    assert eng._fallback_counts["trend"] >= 1, "Expect fallback when requested regime has no index"
    # And RANGE should have zero fallbacks
    assert eng._fallback_counts.get("range", 0) == 0

    # Search log structure
    assert len(eng._search_log) >= 10
    last = eng._search_log[-1]
    assert {"regime_requested","index_used","fallback_depth"} <= set(last), "log must carry regime+index info"
    assert last["regime_requested"] in {"TREND","RANGE","VOL","GLOBAL"}

def test_fallback_rate_computation_small_batch():
    eng = _toy_engine(regimes=["RANGE","VOL","VOL"])
    x = np.array([0.2, 0.05], dtype=np.float32)

    # 10 TREND requests → all should fallback due to no TREND clusters
    n = 10
    for _ in range(n):
        eng.evaluate(x, regime=MarketRegime.TREND)
    fr = eng._fallback_counts["trend"] / max(1, n)
    assert 0.5 <= fr <= 1.0, "Fallback rate should be substantial when regime index is missing"
