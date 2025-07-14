# -------------------------------------------------------------------
# tests/test_safety_integration.py  (NEW)
# -------------------------------------------------------------------
import asyncio, pytest, hypothesis.strategies as st
from hypothesis import given, settings
from execution.core.contracts import SafetyAction
from execution.safety import SafetyFSM, HaltReason

@given(losses=st.lists(st.floats(min_value=-500, max_value=-1), min_size=3, max_size=3))
@settings(max_examples=5)
def test_halt_then_resume(losses):
    q: asyncio.Queue[SafetyAction] = asyncio.Queue()
    fsm = SafetyFSM({"account_equity": 10000, "single_trade_loss_pct": 0.01}, channel=q)
    # inject 3 losers
    for pnl in losses:
        fsm.register_trade(pnl)
    assert not q.empty()
    halt = q.get_nowait()
    assert halt.action == "HALT"
    # inject profit to resume
    fsm.register_trade(2000)
    resumed = q.get_nowait()
    assert resumed.action in ("RESUME", "SIZE_DOWN")