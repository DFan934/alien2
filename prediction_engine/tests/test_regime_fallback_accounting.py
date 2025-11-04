import numpy as np
from pathlib import Path
from prediction_engine.ev_engine import EVEngine
from prediction_engine.market_regime import MarketRegime

def _toy_engine(regime_labels):
    # Minimal valid shapes
    centers = np.array([[0.0, 0.0],
                        [1.0, 0.0],
                        [0.0, 1.0]], dtype=np.float32)
    mu = np.array([+0.01, -0.02, +0.03], dtype=np.float32)
    var = np.array([0.04, 0.05, 0.06], dtype=np.float32)
    var_down = np.array([0.03, 0.04, 0.05], dtype=np.float32)
    # regime per cluster
    cluster_regime = np.array(regime_labels, dtype="U10")
    # fake schema/scales
    schema_len = centers.shape[1]
    scales = np.ones(schema_len, dtype=np.float32)

    eng = EVEngine(
        centers=centers,
        mu=mu,
        var=var,
        var_down=var_down,
        h=1.0,
        metric="euclidean",
        k=3,
        _n_feat_schema=schema_len,
        _scale_vec=scales,
        cluster_regime=cluster_regime,
    )
    return eng

def test_fallback_counts_and_log():
    # No TREND clusters present → must fall back for TREND requests.
    eng = _toy_engine(regime_labels=["RANGE", "RANGE", "VOL"])
    x = np.array([0.2, 0.2], dtype=np.float32)

    # 1) Ask for TREND -> expect fallback (either to RANGE/VOL or GLOBAL)
    eng.evaluate(x, regime=MarketRegime.TREND)
    # 2) Ask for RANGE -> expect direct hit, no fallback
    eng.evaluate(x, regime=MarketRegime.RANGE)

    # Check counters: TREND should have at least one fallback
    assert eng._fallback_counts["trend"] >= 1

    # Check last logged entry contains required fields
    last = eng._search_log[-1]
    assert set(last.keys()) >= {"regime_requested", "index_used", "fallback_depth", "k_eff", "median_d2"}
    # And our info line is being produced (can't capture log easily here, but structure exists)
    assert last["regime_requested"] in ("TREND", "RANGE", "VOL", "GLOBAL")
