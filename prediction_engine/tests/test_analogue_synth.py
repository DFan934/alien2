# ============================================================================
# FILE: tests/test_analogue_synth.py  (NEW)
# ============================================================================
import numpy as np
from prediction_engine.analogue_synth import AnalogueSynth
from prediction_engine.ev_engine import EVEngine, EVResult


def test_beta_non_trivial():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(50, 6)).astype(np.float32)
    mu = rng.normal(scale=0.01, size=50).astype(np.float32)
    var = np.full(50, 0.02, dtype=np.float32)
    var_down = var.copy()

    engine = EVEngine(
        centers=centers,
        mu=mu,
        var=var,
        var_down=var_down,
        h=0.5,
    )
    x = rng.normal(size=6).astype(np.float32)
    res: EVResult = engine.evaluate(x)

    assert (res.beta > 0).sum() > 0, "Beta should contain positive weights"