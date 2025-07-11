# ---------------------------------------------------------------------------
# tests/test_e2e_backtest.py      –   end-to-end smoke back-test
# ---------------------------------------------------------------------------
"""
Minimal end-to-end sanity-check:

1.  Create an EVEngine with two dummy centroids.
2.  Feed 65 synthetic 1-minute bars (the ExecutionManager waits for ≥ 60).
3.  Verify that at least one JSON signal is emitted to the async queue.

Nothing is written to disk; the test runs in < 0.2 s on a laptop.

pytest-only – just `pytest -q tests/test_e2e_backtest.py`.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from prediction_engine.ev_engine import EVEngine
from prediction_engine.execution.manager import ExecutionManager
from prediction_engine.execution.risk_manager import RiskManager


class _DummyLatency:
    """Lat-monitor stub compatible with ExecutionManager (needs `.mean`)."""

    mean: float = 5.0  # ms – arbitrary small latency


@pytest.mark.asyncio
async def test_e2e_signal_generation(tmp_path: Path):
    # ------------------------------------------------------------------
    # 1)  Tiny EVEngine — two 2-D centroids with plausible μ/σ²
    # ------------------------------------------------------------------
    centers = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    mu = np.array([0.02, 0.04], dtype=np.float32)
    var = np.array([0.002, 0.003], dtype=np.float32)
    var_down = var.copy()
    ev = EVEngine(centers=centers, mu=mu, var=var, var_down=var_down, h=1.0)

    # ------------------------------------------------------------------
    # 2)  Risk & Execution managers
    # ------------------------------------------------------------------
    risk = RiskManager(account_equity=100_000.0)
    em = ExecutionManager(
        ev=ev,
        risk_mgr=risk,
        lat_monitor=_DummyLatency(),
        config={},  # default safety config
        log_path=tmp_path / "signals.log",
    )

    # we do NOT start the background writer – the queue is enough
    # await em.start()

    # ------------------------------------------------------------------
    # 3)  Feed 65 synthetic bars so the regime detector is “ready”
    # ------------------------------------------------------------------
    bars = []
    for i in range(65):
        price = 100.0 + 0.25 * i
        bars.append(
            {
                "symbol": "TEST",
                "price": price,
                "adv": 10.0,  # 10 % ADV trade later
                "features": {
                    "f0": float(i),               # dummy numeric col 0
                    "f1": float(i) * 0.1,         # dummy numeric col 1
                    "atr": 0.5,                   # ATR for RiskManager
                },
            }
        )

    # ------------------------------------------------------------------
    # 4)  Drive the async on_bar handler
    # ------------------------------------------------------------------
    for bar in bars:
        await em.on_bar(bar)

    # ------------------------------------------------------------------
    # 5)  Assert that at least ONE trade signal hit the queue
    # ------------------------------------------------------------------
    assert em._sig_q.qsize() > 0, "no trading signal generated end-to-end"

    # clean-up (if you had started .start() earlier)
    # await em.stop()
