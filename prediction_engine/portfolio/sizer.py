# prediction_engine/portfolio/sizer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Dict

@dataclass(frozen=True)
class RiskCaps:
    max_gross_frac: float = 0.10      # at most 10% of capital per trade (dollars)
    adv_cap_pct: float = 0.20         # at most 20% of ADV (shares)
    max_shares: float = 1e9           # hard absolute cap (safety)

def _ramp_weight(p: float, p_gate: float, p_full: float) -> float:
    if p_full <= p_gate:
        return 0.0
    if p <= p_gate:
        return 0.0
    if p >= p_full:
        return 1.0
    return (p - p_gate) / (p_full - p_gate)

def _side_from_p(p: float) -> int:
    return 1 if p >= 0.5 else -1

def _per_share_roundtrip_cost_usd(*, price: float, costs: Dict, adv_frac: float) -> float:
    # components are round-trip
    half_spread_usd = float(costs.get("half_spread_usd", 0.0))
    commission = float(costs.get("commission", 0.0))  # $/share (one way)
    slippage_bp = float(costs.get("slippage_bp", 0.0))
    impact_bps_per_frac = float(costs.get("impact_bps_per_adv_frac", 25.0))

    spread = 2.0 * half_spread_usd
    comm  = 2.0 * commission
    slip  = (slippage_bp / 1e4) * 2.0 * float(price)
    impact= (impact_bps_per_frac * adv_frac / 1e4) * float(price)
    return spread + comm + slip + impact

def size_from_p(
    p_cal: float,
    *,
    vol: float,               # per-bar expected return scale (e.g., sigma)
    capital: float,           # dollars
    risk_caps: RiskCaps,
    costs: Dict,              # must contain 'price'; may include half_spread_usd, commission, etc.
    p_gate: float = 0.55,
    p_full: float = 0.65,
    strategy: Literal["score", "equal_capital", "vol_scaled"] = "score",
    cost_lambda: float = 1.2,  # require EV >= 1.2 × modeled costs
) -> float:
    """
    Returns signed shares (qty). 0 if edge (after lambda) doesn't clear modeled costs.
    EV/share ≈ mu * price, where mu ≈ (2p−1) * vol.
    """
    price = float(costs.get("price", 0.0))
    if price <= 0 or vol <= 0:
        return 0.0

    side = _side_from_p(p_cal)
    score = abs(p_cal - 0.5) * 2.0           # 0..1
    w = _ramp_weight(p_cal, p_gate, p_full)  # 0..1

    # Base dollar target by strategy
    max_dollars = risk_caps.max_gross_frac * capital
    if strategy == "equal_capital":
        target_dollars = w * max_dollars
    elif strategy == "vol_scaled":
        # scale by inverse vol (higher vol → smaller dollar exposure)
        target_dollars = w * max_dollars * min(3.0, 0.02 / vol)
    else:  # "score"
        target_dollars = w * max_dollars * score

    # Convert to shares and cap by ADV / hard cap
    qty_abs = max(0.0, target_dollars / price)
    adv_frac = float(costs.get("adv_frac", 0.0))
    adv_shares = float(costs.get("adv_shares", 0.0))
    if adv_shares > 0:
        qty_abs = min(qty_abs, risk_caps.adv_cap_pct * adv_shares)
    qty_abs = min(qty_abs, risk_caps.max_shares)

    # Cost hurdle: require EV >= lambda * modeled_costs
    mu = (2.0 * p_cal - 1.0) * vol                 # expected return per $1 notional
    ev_per_share_usd = mu * price                  # ≈ per-share expected PnL (USD)
    modeled_cost_usd = _per_share_roundtrip_cost_usd(price=price, costs=costs, adv_frac=adv_frac)
    if ev_per_share_usd < cost_lambda * modeled_cost_usd:
        return 0.0

    return float(side) * qty_abs
