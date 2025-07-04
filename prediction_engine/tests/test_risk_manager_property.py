# ---------------------------------------------------------------------------
# tests/test_risk_manager_property.py
# ---------------------------------------------------------------------------
"""Property‑based tests for ATR‑aware RiskManager sizing.

Minimal required API:
```python
rm = RiskManager(account_equity=100_000, risk_per_trade=0.01)
rm.update_atr("SYM", atr)
qty = rm.position_size("SYM", price, ev_mean, ev_var)
```

We focus on monotonicity properties:
* Higher ATR ⇒ smaller position size
* Higher price ⇒ smaller position (dollar risk constant)
* Increasing account equity ⇒ weakly larger size for fixed ATR & price
"""
from hypothesis import given, settings, strategies as st
import math

try:
    from prediction_engine.execution.risk_manager import RiskManager  # type: ignore
except ModuleNotFoundError:
    import pytest
    pytest.skip("RiskManager not present", allow_module_level=True)

@settings(max_examples=300, deadline=None)
@given(
    atr1=st.floats(0.10, 5.0),
    atr2=st.floats(0.10, 5.0),
    price=st.floats(5.0, 500.0),
    equity=st.floats(50_000, 500_000),
)
def test_position_size_monotone_atr(atr1, atr2, price, equity):
    rm = RiskManager(account_equity=equity, risk_per_trade=0.01)
    rm.update_atr("XYZ", atr1)
    size1 = rm.position_size("XYZ", price, ev_mean=0.01, ev_var=0.02)
    rm.update_atr("XYZ", atr2)
    size2 = rm.position_size("XYZ", price, ev_mean=0.01, ev_var=0.02)

    if atr1 < atr2:
        assert size1 >= size2
    elif atr1 > atr2:
        assert size1 <= size2

@given(
    price1=st.floats(5.0, 500.0),
    price2=st.floats(5.0, 500.0),
    atr=st.floats(0.10, 5.0),
)
@settings(max_examples=200, deadline=None)
def test_position_size_monotone_price(price1, price2, atr):
    rm = RiskManager(account_equity=100_000, risk_per_trade=0.01)
    rm.update_atr("XYZ", atr)
    size1 = rm.position_size("XYZ", price1, ev_mean=0.01, ev_var=0.02)
    size2 = rm.position_size("XYZ", price2, ev_mean=0.01, ev_var=0.02)

    if price1 < price2:
        assert size1 >= size2
    elif price1 > price2:
        assert size1 <= size2

@given(
    equity1=st.floats(50_000, 300_000),
    equity2=st.floats(50_000, 300_000),
    price=st.floats(10.0, 100.0),
    atr=st.floats(0.10, 5.0),
)
@settings(max_examples=200, deadline=None)
def test_position_size_equity(equity1, equity2, price, atr):
    if math.isclose(equity1, equity2):
        return  # avoid flaky equal‑float comparisons
    rm1 = RiskManager(account_equity=equity1, risk_per_trade=0.01)
    rm2 = RiskManager(account_equity=equity2, risk_per_trade=0.01)
    rm1.update_atr("XYZ", atr)
    rm2.update_atr("XYZ", atr)
    size1 = rm1.position_size("XYZ", price, ev_mean=0.02, ev_var=0.03)
    size2 = rm2.position_size("XYZ", price, ev_mean=0.02, ev_var=0.03)
    if equity1 < equity2:
        assert size1 <= size2
    else:
        assert size1 >= size2
