# execution/tests/test_order_policy.py
from __future__ import annotations

from execution.order_policy import OrderPolicyConfig, decision_to_order_request


def test_deterministic_translation_same_inputs_same_client_order_id_and_qty():
    cfg = OrderPolicyConfig(
        max_positions=5,
        max_exposure_usd=10_000.0,
        max_notional_per_symbol_usd=2_000.0,
        max_qty_per_order=500,
        allow_short=False,
    )

    positions = {"MSFT": 10}
    marks = {"MSFT": 300.0}

    req1, r1 = decision_to_order_request(
        cfg=cfg,
        decision_id="dec_0001",
        symbol="AAPL",
        side="BUY",
        desired_qty=10,
        last_price=150.0,
        positions=positions,
        marks=marks,
        reason="test",
        signal_id="sig_x",
    )
    req2, r2 = decision_to_order_request(
        cfg=cfg,
        decision_id="dec_0001",
        symbol="AAPL",
        side="BUY",
        desired_qty=10,
        last_price=150.0,
        positions=positions,
        marks=marks,
        reason="test",
        signal_id="sig_x",
    )

    assert r1 == "ok" and r2 == "ok"
    assert req1 is not None and req2 is not None
    assert req1.client_order_id == req2.client_order_id
    assert req1.qty == req2.qty
    assert req1.symbol == "AAPL"
    assert req1.side == "BUY"


def test_rejects_new_symbol_when_max_positions_reached():
    cfg = OrderPolicyConfig(max_positions=2, max_exposure_usd=100_000.0, max_notional_per_symbol_usd=50_000.0)

    positions = {"AAPL": 1, "MSFT": 1}  # already at 2 symbols open
    marks = {"AAPL": 100.0, "MSFT": 200.0}

    req, reason = decision_to_order_request(
        cfg=cfg,
        decision_id="dec_0002",
        symbol="NVDA",     # new symbol would exceed max_positions
        side="BUY",
        desired_qty=1,
        last_price=500.0,
        positions=positions,
        marks=marks,
    )
    assert req is None
    assert reason == "reject: max_positions"


def test_clips_by_max_exposure_and_per_symbol_notional():
    cfg = OrderPolicyConfig(
        max_positions=5,
        max_exposure_usd=1000.0,              # total cap
        max_notional_per_symbol_usd=300.0,    # per-symbol cap
        max_qty_per_order=10_000,
    )

    # currently holding exposure 800
    positions = {"MSFT": 4}
    marks = {"MSFT": 200.0}  # 4*200 = 800

    # remaining exposure = 200
    # last_price = 50 => remaining exposure allows at most 4 shares
    # but per-symbol cap = 300 => allows at most 6 shares, so remaining exposure dominates => 4
    req, reason = decision_to_order_request(
        cfg=cfg,
        decision_id="dec_0003",
        symbol="AAPL",
        side="BUY",
        desired_qty=100,   # would exceed caps
        last_price=50.0,
        positions=positions,
        marks=marks,
    )

    assert reason == "ok"
    assert req is not None
    assert req.qty == 4


def test_shorts_disabled_rejects_sell_when_no_long_to_reduce():
    cfg = OrderPolicyConfig(allow_short=False, max_positions=5, max_exposure_usd=10_000, max_notional_per_symbol_usd=2_000)

    positions = {"AAPL": 0}  # no long
    marks = {"AAPL": 150.0}

    req, reason = decision_to_order_request(
        cfg=cfg,
        decision_id="dec_0004",
        symbol="AAPL",
        side="SELL",
        desired_qty=1,
        last_price=150.0,
        positions=positions,
        marks=marks,
    )
    assert req is None
    assert reason == "reject: shorts_disabled"
