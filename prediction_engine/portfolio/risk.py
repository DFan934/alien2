# prediction_engine/portfolio/risk.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import pandas as pd

@dataclass
class RiskLimits:
    # dollar-based limits
    max_gross: float                  # e.g., 0.5 * equity
    max_net: float                    # e.g., 0.2 * equity
    per_symbol_cap: float             # dollars, abs
    sector_cap: float                 # dollars, abs per sector
    max_concurrent: int               # open positions
    daily_stop: float                 # negative dollars; stop when day_pnl <= daily_stop
    streak_stop: int = 0              # consecutive losing trades before cooldown (0 = off)

@dataclass
class SafetyFSM:
    halted: bool = False
    cooldown_until: Optional[pd.Timestamp] = None
    loss_streak: int = 0

    def on_trade_pnl(self, pnl: float):
        if pnl < 0:
            self.loss_streak += 1
        else:
            self.loss_streak = 0

class RiskEngine:
    """
    Stateless limit checks + a simple safety FSM. Holds per-sector exposure view you pass in.
    """
    def __init__(self, limits: RiskLimits):
        self.limits = limits
        self.sector_gross: Dict[str, float] = {}

    def _sector_ok(self, sector: Optional[str], add_dollars: float) -> bool:
        if sector is None:
            return True
        after = self.sector_gross.get(sector, 0.0) + abs(add_dollars)
        return after <= self.limits.sector_cap + 1e-9

    def can_open(self, *, symbol: str, sector: Optional[str], qty: float, price: float,
                 now: pd.Timestamp, ledger_snapshot: dict, open_positions: int) -> tuple[bool, str]:
        if qty == 0:
            return False, "zero_qty"

        # daily stop
        if ledger_snapshot.get("day_pnl", 0.0) <= self.limits.daily_stop:
            return False, "daily_stop"

        # concurrency
        if open_positions >= self.limits.max_concurrent and symbol not in ledger_snapshot.get("positions", []):
            return False, "max_concurrent"

        # gross / net after trade
        add_notional = abs(qty * price)
        gross_after = ledger_snapshot["gross"] + add_notional
        side = 1 if qty > 0 else -1
        net_after = ledger_snapshot["net"] + side * add_notional

        if gross_after > self.limits.max_gross + 1e-9:
            return False, "max_gross"
        if abs(net_after) > self.limits.max_net + 1e-9:
            return False, "max_net"

        # per-symbol cap (use avg price proxy)
        sym_notional = ledger_snapshot.get("symbol_notional", {}).get(symbol, 0.0) + add_notional
        if sym_notional > self.limits.per_symbol_cap + 1e-9:
            return False, "per_symbol_cap"

        # sector cap
        if not self._sector_ok(sector, add_notional):
            return False, "sector_cap"

        return True, "ok"

    def on_fill(self, *, symbol: str, sector: Optional[str], notional: float, pnl: float):
        # update sector view
        if sector:
            self.sector_gross[sector] = self.sector_gross.get(sector, 0.0) + abs(notional)
