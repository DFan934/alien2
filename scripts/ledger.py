# portfolio/ledger.py
from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd
from typing import Dict

@dataclass
class Position:
    qty: float = 0.0
    avg_px: float = 0.0

@dataclass
class PortfolioLedger:
    cash: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)

    def apply_fill(self, symbol: str, ts, qty: float, price: float, cost: float = 0.0):
        """
        qty > 0 buy, qty < 0 sell. Updates cash and position.
        """
        self.cash -= qty * price
        self.cash -= cost
        pos = self.positions.get(symbol, Position())
        new_qty = pos.qty + qty
        if new_qty == 0:
            self.positions[symbol] = Position(0.0, 0.0)
        elif pos.qty == 0:
            self.positions[symbol] = Position(new_qty, price)
        else:
            # running average price for same-side inventory
            if (pos.qty > 0 and new_qty > 0) or (pos.qty < 0 and new_qty < 0):
                self.positions[symbol] = Position(
                    new_qty, (pos.avg_px * abs(pos.qty) + price * abs(qty)) / abs(new_qty)
                )
            else:
                # reduced/flip: keep avg_px on remaining; if flip, reset avg_px to last price
                self.positions[symbol] = Position(new_qty, price if (pos.qty * new_qty) < 0 else pos.avg_px)

    def mark_to_market(self, prices: Dict[str, float]) -> float:
        eq = self.cash
        for sym, pos in self.positions.items():
            if pos.qty != 0 and sym in prices:
                eq += pos.qty * prices[sym]
        return eq

def equity_curve_from_trades(trades: pd.DataFrame, price_col: str = "exit_price") -> pd.Series:
    """
    Quick portfolio curve from realized trades only (no open PnL).
    Expects columns: ['exit_ts','realized_pnl'] at minimum.
    """
    if trades.empty:
        return pd.Series(dtype=float)
    pnl = trades.set_index("exit_ts")["realized_pnl"].sort_index()
    return pnl.cumsum()
