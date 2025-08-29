from pathlib import Path

def test_no_utils_timeit_import():
    p = Path("prediction_engine/execution/manager.py")
    src = p.read_text()
    assert "from prediction_engine.utils.latency import timeit" not in src
    assert "from execution.latency import latency_monitor, timeit" in src
