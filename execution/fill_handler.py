# execution/fill_handler.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from execution.contracts_live import OrderUpdateEvent


@dataclass(frozen=True)
class FillEvent:
    fill_id: str
    broker_order_id: str
    client_order_id: str
    symbol: str
    side: str  # "BUY"/"SELL"
    qty: int
    price: float
    event_ts_utc: datetime
    raw: Dict[str, Any]

    def to_row(self) -> Dict[str, Any]:
        d = asdict(self)
        d["event_ts_utc"] = self.event_ts_utc
        return d


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def fill_from_update_delta(
    upd: OrderUpdateEvent,
    *,
    prev_filled_qty: int,
) -> Optional[FillEvent]:
    """
    Convert an OrderUpdateEvent into a FillEvent iff it represents NEW filled quantity
    since prev_filled_qty.

    We rely on:
      - upd.filled_qty (cumulative)
      - upd.last_fill (optional) for price/qty
      - upd.avg_fill_price as fallback price
    """
    new_filled = int(upd.filled_qty or 0)
    prev = int(prev_filled_qty or 0)
    delta = new_filled - prev
    if delta <= 0:
        return None

    # Best-effort price:
    price: Optional[float] = None
    if upd.last_fill is not None and upd.last_fill.price is not None:
        price = float(upd.last_fill.price)
    elif upd.avg_fill_price is not None:
        price = float(upd.avg_fill_price)

    if price is None:
        # If broker didn't provide price, we still emit but set to 0.0 so downstream sees anomaly.
        price = 0.0

    fill_id = (upd.last_fill.fill_id if (upd.last_fill and upd.last_fill.fill_id) else uuid4().hex)

    return FillEvent(
        fill_id=str(fill_id),
        broker_order_id=str(upd.broker_order_id),
        client_order_id=str(upd.client_order_id),
        symbol=str(upd.symbol).upper(),
        side=str(upd.side),
        qty=int(delta),
        price=float(price),
        event_ts_utc=upd.event_ts_utc or _utcnow(),
        raw={"order_update_raw": upd.raw or {}},
    )
