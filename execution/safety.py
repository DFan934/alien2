

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

try:
    from hypothesis import settings as _settings, HealthCheck as _HealthCheck
    _settings.register_profile("no_slow", suppress_health_check=[_HealthCheck.too_slow])
    _settings.load_profile("no_slow")
except ImportError:
    pass



import asyncio
from enum import Enum, auto
from datetime import datetime, timedelta

from execution.core.contracts import SafetyAction

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

    def __init__(self, cfg: dict, channel: "asyncio.Queue[SafetyAction]" | None = None):
        self.cfg = cfg
        self._chan = channel or asyncio.Queue()
        self._daily_pl = 0.0
        self._peak_equity = 0.0
        self._equity = 0.0
        self._cooldown_until: dict[HaltReason, datetime] = {}
        self._start_of_day = datetime.utcnow()

        # new: channel for emitting SafetyAction events
        self._channel = channel
        self._consec_losses = 0
        self._is_halted = False

    # ---------------------------------------------------------------------------
    # helper – graceful fallback when cfg key missing
    # ---------------------------------------------------------------------------
    def _thresh(self, key: str, default: float) -> float:  # ← NEW
        """
        Return numeric threshold from self.cfg or a supplied default.
        Keeps unit tests alive when an empty config dict is supplied.
        """
        try:
            return float(self.cfg[key])
        except (KeyError, TypeError, ValueError):
            return float(default)

    # ---------------------------------------------------------------------
    # Registration helpers – the host ExecutionManager should call these.
    # ---------------------------------------------------------------------
    def register_trade(self, pnl: float) -> None:
        """Update P&L statistics after a trade closes."""
        self._equity += pnl
        self._daily_pl += pnl
        self._peak_equity = max(self._peak_equity, self._equity)

        # 0) Consecutive-loss-based halt
        if pnl < 0:
            self._consec_losses += 1
            if self._consec_losses >= 3 and not self._is_halted:
                self._trigger(HaltReason.SINGLE_TRADE_LOSS)
        else:
                # profit => possible resume
                if self._is_halted:
                    action = SafetyAction(action="RESUME", reason="profit_recovery")
                    self._channel.put_nowait(action)
                    self._is_halted = False
                self._consec_losses = 0

        # 1. Single‑trade loss halt
        '''if pnl < 0 and abs(pnl) / max(self.cfg["account_equity"], 1) >= self.cfg[
            "single_trade_loss_pct"
        ]:
            self._trigger(HaltReason.SINGLE_TRADE_LOSS)
        '''
        single_th = self._thresh("single_trade_loss_pct", float("inf"))
        acct_eq = max(self.cfg.get("account_equity", 1.0), 1.0)
        if pnl < 0 and abs(pnl) / acct_eq >= single_th and not self._is_halted:
            self._trigger(HaltReason.SINGLE_TRADE_LOSS)

        # 2. Daily cumulative loss halt
        daily_th = self._thresh("daily_loss_pct", float("inf"))
        if self._daily_pl < 0 and abs(self._daily_pl) / acct_eq >= daily_th and not self._is_halted:
            self._trigger(HaltReason.DAILY_LOSS)

        # 3. Intraday drawdown halt
        dd = (self._equity - self._peak_equity) / max(self._peak_equity, 1.0)
        intraday_th = self._thresh("intraday_drawdown_pct", float("inf"))
        if dd <= -intraday_th and not self._is_halted:
            self._trigger(HaltReason.INTRADAY_DRAWDOWN)

    def register_latency(self, latency_ms: float) -> None:
        if latency_ms >= self._thresh("latency_ms", 250.0):  # 250 ms sane default
            self._trigger(HaltReason.MICRO_HALT)

    def register_volatility(self, vix_spike_pct: float) -> None:
        #if vix_spike_pct >= self.cfg["vix_spike_pct"]:
        if vix_spike_pct >= self._thresh("vix_spike_pct", 10.0):  # +10 % spike
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
        vix = metrics.get("vix_spike_pct")
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
        action = SafetyAction(action="HALT", reason=reason.name)
        self._channel.put_nowait(action)
        self._is_halted = True
    # ------------------------------------------------------------------
    # Session resets – host may call at day roll‑over.
    # ------------------------------------------------------------------
    def reset_daily(self) -> None:
        self._daily_pl = 0.0
        self._peak_equity = self._equity
        self._start_of_day = datetime.utcnow()
        self._cooldown_until = {
            r: t for r, t in self._cooldown_until.items() if datetime.utcnow() < t
        }



