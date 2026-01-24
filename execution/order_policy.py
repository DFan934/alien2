# execution/order_policy.py
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from execution.contracts_live import OrderRequest


# ------------------------- Config -------------------------

@dataclass(frozen=True)
class OrderPolicyConfig:
    """
    Step 7: deterministic Decision -> OrderRequest translation with invariants.

    Notes:
    - max_positions: max distinct symbols with non-zero position.
    - max_exposure_usd: absolute notional cap across all open positions (gross, simplified).
    - max_notional_per_symbol_usd: per-symbol notional cap for new orders.
    - max_qty_per_order: hard cap to avoid accidental large orders.
    """
    max_positions: int = 5
    max_exposure_usd: float = 10_000.0
    max_notional_per_symbol_usd: float = 2_000.0
    max_qty_per_order: int = 500
    allow_short: bool = False  # BUY-only unless you explicitly allow shorts


# ------------------------- Helpers -------------------------

def _stable_client_order_id(decision_id: str, symbol: str, side: str) -> str:
    """
    Deterministic idempotency key. No randomness -> testable determinism.
    """
    s = f"{decision_id}|{symbol.upper()}|{side.upper()}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:24]


def _count_open_positions(positions: Dict[str, int]) -> int:
    return sum(1 for _, q in positions.items() if int(q) != 0)


def _gross_exposure_usd(positions: Dict[str, int], marks: Dict[str, float]) -> float:
    """
    Simplified gross exposure: sum(|qty| * mark). Requires marks for held symbols.
    Missing marks treated as 0 (conservative for gating in tests).
    """
    exp = 0.0
    for sym, qty in positions.items():
        q = int(qty)
        if q == 0:
            continue
        px = float(marks.get(sym.upper(), 0.0))
        exp += abs(q) * px
    return exp


def _clip_qty_by_notional(qty: int, price: float, notional_cap: float) -> int:
    if price <= 0:
        return 0
    max_qty = int(notional_cap // price)
    return max(0, min(int(qty), max_qty))


# ------------------------- Main API -------------------------

def decision_to_order_request(
    *,
    cfg: OrderPolicyConfig,
    decision_id: str,
    symbol: str,
    side: str,
    desired_qty: int,
    last_price: float,
    positions: Dict[str, int],
    marks: Dict[str, float],
    reason: Optional[str] = None,
    signal_id: Optional[str] = None,
) -> Tuple[Optional[OrderRequest], str]:
    """
    Pure function:
      Decision inputs -> (OrderRequest or None, status_reason)

    Invariants enforced:
      - side must be BUY/SELL, with allow_short gating
      - max_positions (distinct symbols) gating
      - max_exposure_usd gating (gross)
      - per-symbol notional cap
      - deterministic client_order_id
      - qty > 0, qty <= max_qty_per_order
    """
    sym = symbol.upper().strip()
    sd = side.upper().strip()

    if sd not in ("BUY", "SELL"):
        return None, "reject: invalid_side"

    if sd == "SELL" and not cfg.allow_short and int(positions.get(sym, 0)) <= 0:
        # if you don't allow shorting, SELL is only allowed to reduce an existing long
        return None, "reject: shorts_disabled"

    if desired_qty <= 0:
        return None, "reject: qty_non_positive"

    if last_price <= 0:
        return None, "reject: bad_price"

    # max_positions gating: if opening a NEW symbol position
    open_positions = _count_open_positions(positions)
    is_new_symbol = int(positions.get(sym, 0)) == 0
    if is_new_symbol and open_positions >= int(cfg.max_positions):
        return None, "reject: max_positions"

    # exposure gating (gross): current + proposed <= cap
    current_exp = _gross_exposure_usd(positions, marks)
    proposed_notional = float(desired_qty) * float(last_price)
    if current_exp >= float(cfg.max_exposure_usd):
        return None, "reject: max_exposure_reached"

    remaining_exp = float(cfg.max_exposure_usd) - current_exp
    # clip by remaining exposure
    qty1 = _clip_qty_by_notional(int(desired_qty), float(last_price), remaining_exp)

    # clip by per-symbol notional
    qty2 = _clip_qty_by_notional(qty1, float(last_price), float(cfg.max_notional_per_symbol_usd))

    # clip by max_qty_per_order
    qty3 = min(int(qty2), int(cfg.max_qty_per_order))

    if qty3 <= 0:
        return None, "reject: clipped_to_zero"

    req = OrderRequest(
        client_order_id=_stable_client_order_id(decision_id, sym, sd),
        symbol=sym,
        side=sd,          # normalized by contract too :contentReference[oaicite:1]{index=1}
        qty=int(qty3),
        order_type="MKT",
        tif="DAY",
        signal_id=signal_id,
        reason=reason or "live_decision",
    )
    return req, "ok"
