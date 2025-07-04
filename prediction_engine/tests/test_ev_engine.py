# ---------------------------------------------------------------------------
# tests/test_ev_engine.py (NEW minimal test)
# ---------------------------------------------------------------------------
import numpy as np
from prediction_engine.ev_engine import EVEngine


def test_ev_monotonic():
    centers = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32)
    stats = {0: (0.01, 0.001), 1: (0.02, 0.002), 2: (0.03, 0.003)}
    ev = EVEngine(centers, stats)
    e1 = ev.evaluate(np.array([0, 0]), 1000, 1e6).expected
    e2 = ev.evaluate(np.array([2, 2]), 1000, 1e6).expected
    assert e2 > e1
