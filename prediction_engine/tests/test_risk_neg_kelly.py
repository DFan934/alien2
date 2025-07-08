from ..execution.risk_manager import RiskManager

def test_negative_kelly_clamp():
    rm = RiskManager(account_equity=10_000)
    qty = rm.kelly_position(mu=-0.01, variance_down=0.02, price=100)
    assert qty == 0
