# prediction_engine/portfolio/ledger.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import pandas as pd

@dataclass
class Position:
    qty: float = 0.0
    avg_price: float = 0.0  # dollar-weighted average
    sector: Optional[str] = None

@dataclass
class TradeFill:
    symbol: str
    ts: pd.Timestamp
    side: int           # +1 buy, -1 sell
    qty: float
    price: float
    fees: float = 0.0
    sector: Optional[str] = None

@dataclass
class PortfolioLedger:
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    gross: float = 0.0
    net: float = 0.0
    realized_pnl: float = 0.0
    day_pnl: float = 0.0
    day: Optional[pd.Timestamp] = None  # UTC date boundary for daily stop

    def _roll_day_if_needed(self, ts: pd.Timestamp):
        d = pd.Timestamp(ts).tz_convert("UTC").normalize()
        if self.day is None or d != self.day:
            self.day = d
            self.day_pnl = 0.0

    def on_fill(self, fill: TradeFill):
        self._roll_day_if_needed(fill.ts)
        pos = self.positions.get(fill.symbol, Position(sector=fill.sector))

        # cash impact
        cash_delta = -fill.side * fill.qty * fill.price - fill.fees
        self.cash += cash_delta

        # realized PnL if closing or reducing
        realized = 0.0
        if pos.qty != 0 and (pos.qty > 0) != (fill.side > 0):
            close_qty = min(abs(pos.qty), fill.qty)
            realized = (fill.price - pos.avg_price) * (1 if pos.qty > 0 else -1) * close_qty
            self.realized_pnl += realized
            self.day_pnl += realized

        # position update (simple WAC)
        new_qty = pos.qty + fill.side * fill.qty
        if new_qty == 0:
            pos.qty, pos.avg_price = 0.0, 0.0
        elif (pos.qty >= 0 and new_qty >= 0) or (pos.qty <= 0 and new_qty <= 0):
            # same direction → update avg
            notional_old = abs(pos.qty) * pos.avg_price
            notional_new = fill.qty * fill.price
            pos.avg_price = (notional_old + notional_new) / max(abs(new_qty), 1e-12)
            pos.qty = new_qty
        else:
            # flipped sign → avg becomes trade price
            pos.qty = new_qty
            pos.avg_price = fill.price

        self.positions[fill.symbol] = pos
        self._recompute_exposure()

    def _recompute_exposure(self):
        # gross/net exposure in shares*price terms requires a price snapshot; in backtests we
        # approximate using avg_price (conservative). If you have a live mid, pass it here instead.
        gross = 0.0
        net = 0.0
        for p in self.positions.values():
            gross += abs(p.qty * p.avg_price)
            net += p.qty * p.avg_price
        self.gross = gross
        self.net = net

    def snapshot_row(self) -> dict:
        return {
            "cash": float(self.cash),
            "realized_pnl": float(self.realized_pnl),
            "day_pnl": float(self.day_pnl),
            "gross": float(self.gross),
            "net": float(self.net),
        }
