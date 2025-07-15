import pytest
from execution.risk_manager import RiskManager

@pytest.fixture
def rm():
    # start with $1,000 equity
    return RiskManager(account_equity=1000.0)

def test_buy_increases_position_and_equity(rm):
    is_closed, pnl, tid = rm.process_fill({'price': 10.0, 'size': 5.0, 'side': 'buy', 'trade_id': 't1'})
    assert not is_closed
    assert pnl == pytest.approx(0.0)
    assert rm.position_size == pytest.approx(5.0)
    assert rm.avg_entry_price == pytest.approx(10.0)
    assert rm.account_equity == pytest.approx(1000.0)

def test_partial_sell_reduces_position_and_realizes_pnl(rm):
    # enter first
    rm.process_fill({'price': 10.0, 'size': 5.0, 'side': 'buy', 'trade_id': 't1'})
    # partial exit
    is_closed, pnl, tid = rm.process_fill({'price': 12.0, 'size': 2.0, 'side': 'sell', 'trade_id': 't1'})
    assert not is_closed
    assert pnl == pytest.approx((12.0 - 10.0) * 2.0)
    assert rm.position_size == pytest.approx(3.0)
    assert rm.account_equity == pytest.approx(1000.0 + pnl)

def test_full_sell_resets_position_and_entry(rm):
    rm.process_fill({'price': 10.0, 'size': 5.0, 'side': 'buy', 'trade_id': 't2'})
    is_closed, pnl, tid = rm.process_fill({'price': 8.0, 'size': 5.0, 'side': 'sell', 'trade_id': 't2'})
    assert is_closed
    assert pnl == pytest.approx((8.0 - 10.0) * 5.0)
    assert rm.position_size == pytest.approx(0.0)
    assert rm.avg_entry_price == pytest.approx(0.0)

def test_drawdown_tracking(rm):
    # lose money
    rm.process_fill({'price': 10.0, 'size': 5.0, 'side': 'buy', 'trade_id': 't3'})
    _, pnl1, _ = rm.process_fill({'price': 8.0, 'size': 5.0, 'side': 'sell', 'trade_id': 't3'})
    assert rm.max_drawdown == pytest.approx(abs(pnl1))
    # win money afterwards
    rm.process_fill({'price': 9.0, 'size': 2.0, 'side': 'buy', 'trade_id': 't4'})
    _, pnl2, _ = rm.process_fill({'price': 12.0, 'size': 2.0, 'side': 'sell', 'trade_id': 't4'})
    assert rm.account_equity == pytest.approx(1000.0 + pnl1 + pnl2)
    # drawdown stays at the worst observed level
    assert rm.max_drawdown >= abs(pnl1)



import pytest
from execution.risk_manager import RiskManager

def test_process_fill_full_cycle():
    rm = RiskManager(account_equity=1000.0)
    # entry fill
    closed, pnl, tid = rm.process_fill({
        "trade_id":"T1","symbol":"AAPL","price":100.0,"size":10.0,"side":"buy"
    })
    assert not closed and pnl == 0
    # exit fill
    closed2, pnl2, _ = rm.process_fill({
        "trade_id":"T1","symbol":"AAPL","price":110.0,"size":10.0,"side":"sell"
    })
    assert closed2 and pytest.approx(pnl2) == (110.0-100.0)*10
    # equity updated
    assert rm.account_equity == pytest.approx(1000.0 + pnl2)

