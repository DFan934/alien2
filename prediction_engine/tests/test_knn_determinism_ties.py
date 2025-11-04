import numpy as np
from prediction_engine.distance_calculator import DistanceCalculator

def test_knn_order_is_deterministic_on_ties():
    # Two centers equidistant from x
    ref = np.array([[+1.0, 0.0],
                    [-1.0, 0.0]], dtype=np.float32)
    x   = np.array([0.0, 0.0], dtype=np.float32)

    dc = DistanceCalculator(ref, metric="euclidean")
    # Run multiple times; order must be stable (idx=[0,1] OR [1,0], but same every time)
    runs = []
    for _ in range(5):
        d2, idx = dc.batch_top_k(x[np.newaxis,:], k=2)
        runs.append(idx[0].tolist())

    assert all(r == runs[0] for r in runs), f"non-deterministic order: {runs}"
