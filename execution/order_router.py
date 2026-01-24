# execution/order_router.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from execution.contracts_live import OrderUpdateEvent
from execution.fill_handler import FillEvent, fill_from_update_delta


@dataclass
class _OrderState:
    filled_qty: int = 0


class OrderRouter:
    """
    Tracks per-order cumulative filled_qty and emits FillEvent for *delta* fills.
    Deterministic: given the same update sequence, you get the same FillEvents.
    """
    def __init__(self) -> None:
        self._by_broker_id: Dict[str, _OrderState] = {}

    def on_order_update(self, upd: OrderUpdateEvent) -> List[FillEvent]:
        bid = str(upd.broker_order_id)
        st = self._by_broker_id.get(bid)
        if st is None:
            st = _OrderState(filled_qty=0)
            self._by_broker_id[bid] = st

        fe = fill_from_update_delta(upd, prev_filled_qty=st.filled_qty)

        # always advance cumulative
        st.filled_qty = int(upd.filled_qty or 0)

        return [] if fe is None else [fe]
