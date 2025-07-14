from execution.latency import latency_monitor, timeit
import time, random

@timeit("foo")
def fake_work():
    time.sleep(random.uniform(0.001, 0.004))

def test_p95():
    for _ in range(300):
        fake_work()
    assert 1.0 <= latency_monitor.p95("foo") <= 6.0
