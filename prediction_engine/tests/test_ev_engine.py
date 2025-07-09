# ---------------------------------------------------------------------------
# tests/test_ev_engine.py (NEW minimal test)
# ---------------------------------------------------------------------------
import numpy as np
from prediction_engine.ev_engine import EVEngine

def test_ev_monotonic():
    centers = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32)
    mu = np.array([0.01, 0.02, 0.03], dtype=np.float32)
    var = np.array([0.001, 0.002, 0.003], dtype=np.float32)
    var_down = np.array([0.0005, 0.001, 0.0015], dtype=np.float32)
    h = 1.0

    ev = EVEngine(
        centers=centers,
        mu=mu,
        var=var,
        var_down=var_down,
        h=h,
    )
    e1 = ev.evaluate(np.array([0, 0], dtype=np.float32)).mu
    e2 = ev.evaluate(np.array([2, 2], dtype=np.float32)).mu
    assert e2 > e1
