import asyncio
from pathlib import Path

import numpy as np
import types
import pytest

from execution.risk_manager import RiskManager
from execution.manager import ExecutionManager

class StubLatency:
    def __init__(self):
        self.calls = []
    def mean(self, label: str) -> float:
        self.calls.append(label)
        return 0.0

class DummyEV:
    def __init__(self, n_features=1):
        self.centers = np.zeros((4, n_features), dtype=np.float32)
    def feature_names(self):
        return [f"f{i}" for i in range(self.centers.shape[1])]
    def evaluate(self, x, adv_percentile=None, regime=None):
        # returns a lightweight object with required fields
        return types.SimpleNamespace(
            mu=0.001,
            sigma=0.01,
            variance_down=0.01,
            cluster_id=0,
            outcome_probs={}
        )

def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)

def test_start_calls_latency_mean_label(tmp_path):
    lat = StubLatency()
    ev = DummyEV(1)
    rm = RiskManager(account_equity=10_000.0)
    em = ExecutionManager(
        ev=ev,
        risk_mgr=rm,
        lat_monitor=lat,
        config={"max_queue": 64},
        log_path=tmp_path / "signals.log",
    )
    run(em.start())
    assert "execution_bar" in lat.calls
    run(em.stop())

def test_safetyfsm_instantiated_once(monkeypatch, tmp_path):
    # count instances
    constructed = {"n": 0}
    from execution import safety as safety_mod

    class _SFake:
        def __init__(self, *args, **kwargs):
            constructed["n"] += 1

    monkeypatch.setattr(safety_mod, "SafetyFSM", _SFake, raising=True)

    lat = StubLatency()
    ev = DummyEV(1)
    rm = RiskManager(account_equity=10_000.0)
    _ = ExecutionManager(
        ev=ev,
        risk_mgr=rm,
        lat_monitor=lat,
        config={"max_queue": 64},
        log_path=tmp_path / "signals.log",
    )
    assert constructed["n"] == 1, "SafetyFSM should be constructed exactly once"
