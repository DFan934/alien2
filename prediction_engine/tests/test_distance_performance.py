# ----------------------------------------------------
# tests/test_distance_performance.py  (optional heavy)
# ----------------------------------------------------
"""Performance benchmark – requires FAISS and pytest-benchmark plugin.

The *benchmark* fixture returns the **result of the timed function**, *not* the
BenchmarkStats object.  To access timing info we must query
``benchmark.stats.stats['mean']`` after the call.
"""
import numpy as np
import pytest

try:
    import faiss  # noqa: F401
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from prediction_engine.distance_calculator import DistanceCalculator

pytestmark = pytest.mark.perf


@pytest.mark.skipif(not HAS_FAISS, reason="FAISS not installed – perf test skipped")
def test_speed_1m32(benchmark):
    np.random.seed(0)
    ref = np.random.randn(1_000_000, 32).astype(np.float32)
    calc = DistanceCalculator(ref, metric="euclidean", ann_backend="faiss")
    Q = np.random.randn(1, 32).astype(np.float32)

    # measure *search* latency only – build time excluded
    benchmark(lambda: calc.batch_top_k(Q, k=32))
    mean_runtime = benchmark.stats.stats["mean"]
    assert mean_runtime < 0.12


def test_speed_1m32(benchmark):
    if not HAS_FAISS:
        pytest.skip("FAISS not installed – perf test skipped")
    np.random.seed(0)
    ref = np.random.randn(1_000_000, 32).astype(np.float32)

    calc = DistanceCalculator(ref, metric="euclidean", ann_backend="faiss")
    Q = np.random.randn(1, 32).astype(np.float32)

    def _search():
        calc.batch_top_k(Q, k=32)

    result = benchmark(_search)
    assert result.stats.stats.mean < 0.12, "Search slower than 120 ms"
