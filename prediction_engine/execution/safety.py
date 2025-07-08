

from __future__ import annotations

"""Five‑Tier Safety Finite‑State Machine (FSM)
============================================

This module implements the *adaptive* safety mechanism described in the BO⭑BBY
blueprint (section 4).  It is designed to be imported by
``execution/manager.py`` and queried *before every order*.

Usage
-----
```python
from execution.safety import SafetyFSM, HaltReason

safety = SafetyFSM(config)
...
if safety.should_halt(latency_ms, symbol_volatility):
    return  # skip trade
...
safety.register_trade(pnl, risk_perc)
```

Config keys (can come from yaml or env):
* **single_trade_loss_pct** – max % loss per trade.
* **daily_loss_pct** – max cumulative daily loss.
* **intraday_drawdown_pct** – max peak‑to‑trough drawdown since session start.
* **latency_ms_threshold** – average latency halt.
* **vix_spike_pct** – intraday VIX spike to trigger halt.
* **cooldowns** – dict mapping HaltReason → seconds.

All percentages are *absolute* (i.e. 0.05 == 5 %).
"""

from enum import Enum, auto
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

__all__ = ["SafetyFSM", "HaltReason"]

COOLDOWNS = {
    "MICRO_HALT": 30,
    "SINGLE_TRADE_LOSS": 900,
    "DAILY_LOSS": 28_800,  # 8 h
    "INTRADAY_DRAWDOWN": 7_200,  # 2 h
    "VOLATILITY_HALT": 900,
}


class HaltReason(Enum):
    """Enumeration of possible safety stops."""

    MICRO_HALT = auto()  # latency spike or VIX spike
    SINGLE_TRADE_LOSS = auto()
    DAILY_LOSS = auto()
    INTRADAY_DRAWDOWN = auto()
    VOLATILITY_HALT = auto()



class SafetyFSM:
    """Adaptive safety guard that can *halt* trading under five conditions.

    The FSM maintains internal counters and timestamps, exposing
    :meth:`should_halt` for quick checks inside order loops.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._daily_pl = 0.0
        self._peak_equity = 0.0
        self._equity = 0.0
        self._cooldown_until: dict[HaltReason, datetime] = {}
        self._start_of_day = datetime.utcnow()

    # ---------------------------------------------------------------------
    # Registration helpers – the host ExecutionManager should call these.
    # ---------------------------------------------------------------------
    def register_trade(self, pnl: float) -> None:
        """Update P&L statistics after a trade closes."""
        self._equity += pnl
        self._daily_pl += pnl
        self._peak_equity = max(self._peak_equity, self._equity)

        # 1. Single‑trade loss halt
        if pnl < 0 and abs(pnl) / max(self.cfg["account_equity"], 1) >= self.cfg[
            "single_trade_loss_pct"
        ]:
            self._trigger(HaltReason.SINGLE_TRADE_LOSS)

        # 2. Daily cumulative loss halt
        if (
            self._daily_pl < 0
            and abs(self._daily_pl) / max(self.cfg["account_equity"], 1)
            >= self.cfg["daily_loss_pct"]
        ):
            self._trigger(HaltReason.DAILY_LOSS)

        # 3. Intraday drawdown halt
        dd = (self._equity - self._peak_equity) / max(self._peak_equity, 1)
        if dd <= -self.cfg["intraday_drawdown_pct"]:
            self._trigger(HaltReason.INTRADAY_DRAWDOWN)

    def register_latency(self, latency_ms: float) -> None:
        if latency_ms >= self.cfg["latency_ms"]:
            self._trigger(HaltReason.MICRO_HALT)

    def register_volatility(self, vix_spike_pct: float) -> None:
        if vix_spike_pct >= self.cfg["vix_spike_pct"]:
            self._trigger(HaltReason.VOLATILITY_HALT)

        # ------------------------------------------------------------------
        # Public query (re-worked to take **metrics)
        # ------------------------------------------------------------------

    def should_halt(self, **metrics) -> bool:
        """
        Quick check before every order.

        Accepts any keyword metrics.  Currently inspected keys
        -------------------------------------------------------
        latency_ms       : float
        vix_spike_pct    : float
        """
        lat = metrics.get("latency_ms")
        vix = metrics.get("vix_spike_pct") or metrics.get("volatility")

        if lat is not None:
            self.register_latency(lat)
        if vix is not None:
            self.register_volatility(vix)

        now = datetime.utcnow()
        return any(now < until for until in self._cooldown_until.values())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _trigger(self, reason: HaltReason) -> None:
        duration = timedelta(seconds=COOLDOWNS.get(reason.name, 300))
        self._cooldown_until[reason] = datetime.utcnow() + duration

    # ------------------------------------------------------------------
    # Session resets – host may call at day roll‑over.
    # ------------------------------------------------------------------
    def reset_daily(self) -> None:
        self._daily_pl = 0.0
        self._peak_equity = self._equity
        self._start_of_day = datetime.utcnow()
        # Clear all halted states that were based on previous day’s stats
        self._cooldown_until = {
            r: t for r, t in self._cooldown_until.items() if datetime.utcnow() < t
        }
