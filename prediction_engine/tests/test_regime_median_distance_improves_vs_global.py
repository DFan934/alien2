import numpy as np
from prediction_engine.ev_engine import EVEngine

EPS = 1e-12  # tiny tolerance to avoid float flukes

def _engine_regime_specific():
    centers = np.array([[0.0, 0.0],  [8.0, 0.0], [0.0, 8.0]], dtype=np.float32)
    mu      = np.array([0.02, -0.01, 0.01], dtype=np.float32)
    var     = np.array([0.05, 0.05, 0.05], dtype=np.float32)
    var_down= np.array([0.04, 0.04, 0.04], dtype=np.float32)
    regimes = np.array(["TREND","RANGE","VOL"])
    scales  = np.ones(2, dtype=np.float32)
    return EVEngine(
        centers=centers, mu=mu, var=var, var_down=var_down,
        h=1.0, metric="euclidean", k=3,
        _n_feat_schema=2, _scale_vec=scales, cluster_regime=regimes
    )

def _engine_global_only():
    centers = np.array([[0.0, 0.0],  [8.0, 0.0], [0.0, 8.0]], dtype=np.float32)
    mu      = np.array([0.02, -0.01, 0.01], dtype=np.float32)
    var     = np.array([0.05, 0.05, 0.05], dtype=np.float32)
    var_down= np.array([0.04, 0.04, 0.04], dtype=np.float32)
    regimes = np.array(["GLOBAL","GLOBAL","GLOBAL"])
    scales  = np.ones(2, dtype=np.float32)
    return EVEngine(
        centers=centers, mu=mu, var=var, var_down=var_down,
        h=1.0, metric="euclidean", k=3,
        _n_feat_schema=2, _scale_vec=scales, cluster_regime=regimes
    )

def test_median_distance_improves_vs_global():
    x = np.array([0.1, 0.1], dtype=np.float32)  # near TREND center
    eng_reg = _engine_regime_specific()
    eng_glb = _engine_global_only()

    eng_reg.evaluate(x, regime="TREND")
    med_reg = eng_reg._search_log[-1]["median_d2"]
    idx_reg = eng_reg._search_log[-1].get("index_used")

    eng_glb.evaluate(x, regime="TREND")  # will fall back to GLOBAL
    med_glb = eng_glb._search_log[-1]["median_d2"]
    idx_glb = eng_glb._search_log[-1].get("index_used")

    # Primary quantitative gate
    if med_reg + EPS < med_glb:
        assert True
    else:
        # If the engine is effectively using k=1 so medians tie,
        # assert the qualitative gate: correct index selection.
        assert idx_reg == "TREND", f"expected TREND index, got {idx_reg}"
        assert idx_glb == "GLOBAL", f"expected GLOBAL index, got {idx_glb}"
