# ---------------------------------------------------------------------------
# tests/test_execution.py
# ---------------------------------------------------------------------------
import asyncio
import math
from datetime import datetime

import pytest

from execution.manager import ExecutionManager
from execution.core.contracts import TradeSignal

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def make_signal(
    symbol: str,
    side: str,
    price: float,
    atr: float,
    confidence: float = 0.8,
):
    """Factory to keep tests concise."""
    return TradeSignal(
        symbol=symbol,
        side=side,           # "BUY" or "SELL"
        price=price,
        atr=atr,
        confidence=confidence,
        timestamp=datetime.utcnow(),
        vwap_dist=0.0,
        ema_fast_dist=0.0,
        orderflow_delta=0.0,
        regime="trend",
    )

# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_position_size_and_stop_levels():
    exec_mgr = ExecutionManager(equity=10_000)

    sig = make_signal("AAPL", "BUY", price=100.0, atr=2.0)
    await exec_mgr.handle_signal(sig)

    pos = exec_mgr.store.get(sig.id)          # ←  uses PositionStore
    assert pos is not None, "Position not created"

    expected_stop = 100.0 - 1.5 * 2.0         # BUY side ⇒ stop below entry
    assert math.isclose(pos[5], expected_stop, rel_tol=1e-6)  # stop_px col

@pytest.mark.asyncio
async def test_atr_zero_edge():
    exec_mgr = ExecutionManager(equity=10_000)
    sig = make_signal("MSFT", "BUY", price=100.0, atr=0.0)

    with pytest.raises(ValueError):
        await exec_mgr.handle_signal(sig)

@pytest.mark.asyncio
async def test_equity_zero_edge():
    exec_mgr = ExecutionManager(equity=0.0)
    sig = make_signal("GOOG", "BUY", price=100.0, atr=1.5)

    with pytest.raises(ValueError):
        await exec_mgr.handle_signal(sig)
