# -----------------------------------------------------------------------------
# execution/risk_manager.py
# -----------------------------------------------------------------------------
"""ATR‑aware Kelly risk sizing + drawdown tracking (M1)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from prediction_engine.tx_cost import BasicCostModel

@dataclass
class RiskManager:
    account_equity: float
    # ─── RE‑ENABLED BRAKES ───────────────────────────────────────────    max_leverage: float = 2.0
    max_leverage: float = 2.0 # notional / equity cap
    max_kelly: float = 0.5 # Kelly fraction hard‑cap
    adv_cap_pct: float = 0.20 # %‑of‑ADV liquidity cap
    cost_model: Optional[BasicCostModel] = None
    risk_per_trade: float = 0.001  # 0.1 %
    atr_multiplier: float = 1.5
    safety_fsm: "Optional[object]" = None  # lazy import

    # ── for process_fill PnL bookkeeping ────────────────────────────────────
    position_size: float = field(default=0.0, init=False)
    avg_entry_price: float = field(default=0.0, init=False)
    max_drawdown: float = field(default=0.0, init=False)

    _symbol_atr: Dict[str, float] = field(default_factory=dict, init=False)
    _peak_equity: float = field(init=False)



    _open_positions: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False)


    # --------------------- lifecycle ----------------------------------------
    def __post_init__(self):
        self._peak_equity = self.account_equity
        # track live PnL state
        self.position_size: float = 0.0
        self.avg_entry_price: float = 0.0
        self.max_drawdown: float = 0.0

    # --------------------- real‑time updates ---------------------------------
    def update_atr(self, symbol: str, atr: float):
        if atr and atr > 0:
            self._symbol_atr[symbol] = atr

    # --------------------- sizing helpers ------------------------------------
    def _dollar_risk(self) -> float:
        return self.account_equity * self.risk_per_trade

    def desired_size(self, symbol: str, price: float) -> int:
        if price <= 0 or math.isnan(price):
            return 0
        atr = self._symbol_atr.get(symbol)
        stop_dist = max(self.atr_multiplier * atr, 0.01 * price) if atr else 0.01 * price
        qty = math.floor(self._dollar_risk() / max(stop_dist, 1e-6))
        max_qty = math.floor(self.account_equity * self.max_leverage / price)
        return max(0, min(qty, max_qty))

    def kelly_position(self,
                       mu: float,
                       variance_down: float,
                       price: float,
                       adv: float | None = None,) -> int:
        # --- 1. SAFEGUARDS FIRST ---
        # Exit immediately if any inputs are invalid for the formula.
        # A non-positive mu means no expected profit.
        # A near-zero variance is mathematically invalid for this formula.
        # ── 1) SAFEGUARDS FIRST ────────────────────────────────────────────
        import math, logging
        # never size if equity is already corrupted
        if not math.isfinite(self.account_equity) or self.account_equity <= 0:
            logging.warning("kelly_position bail: bad account_equity=%r", self.account_equity)
            return 0
        #if price <= 0 or variance_down <= 1e-9 or mu <= 0:
        #    return 0

        # Bail only on truly invalid inputs.  Tiny variance is *expected* now that
        # labels are 1-bar log returns; floor it instead of zero-sizing.
        if price <= 0 or not math.isfinite(variance_down):
            return 0
        var_eff = max(variance_down, 1e-8)  # <-- safety floor (tune)
        kelly_f = min(mu / (2 * var_eff), self.max_kelly)

        # --- 2. CALCULATIONS (only if inputs are safe) ---
        # Now that inputs are validated, proceed with the Kelly formula.
        #kelly_f = min(mu / (2 * variance_down), self.max_kelly)
        dollar_notional = kelly_f * self.account_equity
        max_notional = self.account_equity * self.max_leverage

        #qty = math.floor(min(dollar_notional, max_notional) / price)

        raw_qty = min(dollar_notional, max_notional) / price
        # guard against any non‐finite result
        if not math.isfinite(raw_qty):
            logging.warning(
            "kelly raw_qty non‐finite: %r (dollar_notional=%r, max_notional=%r, price=%r)",
            raw_qty, dollar_notional, max_notional, price
            )
            return 0
        qty = math.floor(raw_qty)

        # --- 3. APPLY LIQUIDITY BRAKE ---
        if adv is not None and adv > 0:
            qty = min(qty, math.floor(adv * self.adv_cap_pct))

        # Ensure final quantity is not negative
        return max(qty, 0)

    # --------------------- drawdown tracking ---------------------------------
    def on_closed_trade(self, pnl: float):
        self.account_equity += pnl
        self._peak_equity = max(self._peak_equity, self.account_equity)
        if self.safety_fsm is not None:
            self.safety_fsm.register_trade(pnl)

    def drawdown(self) -> float:
        return 1.0 - self.account_equity / self._peak_equity

    @staticmethod
    def scale_variance(var: float, adv_percentile: float | None) -> float:
        if adv_percentile is None:
            return var
        mult = 1.0 + 0.5 * min(max((adv_percentile - 5) / 15, 0.0), 1.0)
        return var * mult



    # --------------------------------------------------------------------
    # Value-at-risk helper (for SafetyFSM & metrics)
    # --------------------------------------------------------------------
    def position_value_at_risk(self, entry_px: float, stop_px: float, qty: int) -> float:
        """Absolute dollar risk of an open position."""
        return abs(entry_px - stop_px) * qty





    def process_fill(
        self,
        fill: dict | float,
        fill_size: float | None = None,
        fill_side: str | None = None,
        trade_id: str | None = None,
    ) -> tuple[bool, float, str]:
        """
        Handle a fill event and update position, PnL, equity & drawdown.

        Accept either:
          • fill dict with keys 'price','size','side','trade_id'
          • legacy args (fill_price, fill_size, fill_side[, trade_id])

        Returns:
          (is_trade_closed, realized_pnl, trade_id)
        """
        # unpack dict vs. positional
        if isinstance(fill, dict):
            data = fill
            price = float(data["price"])
            size  = float(data["size"])
            side  = data["side"]
            tid   = data.get("trade_id", "")
        else:
            price = float(fill)
            size  = float(fill_size or 0.0)
            side  = fill_side or ""
            tid   = trade_id or ""

        realized_pnl = 0.0
        # BUY increases position
        if side.lower() == "buy":
            total_cost = self.avg_entry_price * self.position_size + price * size
            self.position_size += size
            self.avg_entry_price = total_cost / self.position_size
        # SELL closes position
        elif side.lower() == "sell":
            if self.position_size <= 0:
                raise ValueError("No position to sell")
            realized_pnl = (price - self.avg_entry_price) * size
            self.position_size -= size
            if self.position_size <= 0:
                self.position_size    = 0.0
                self.avg_entry_price = 0.0
        else:
            raise ValueError(f"Unknown fill_side: {side!r}")

        # update equity & drawdown
        #self.account_equity += realized_pnl
        # --- deduct cost ----------------------------------------------------
        trade_cost = self.cost_model.cost(qty=size) if self.cost_model else 0.0
        self.account_equity += realized_pnl - trade_cost
        self._peak_equity = max(self._peak_equity, self.account_equity)
        drawdown = self._peak_equity - self.account_equity
        self.max_drawdown = max(self.max_drawdown, drawdown)

        # in process_fill, after you compute `realized_pnl`:
        self.last_loss = max(0.0, -realized_pnl)
        self.day_pl = getattr(self, "day_pl", 0.0) + realized_pnl

        is_closed = (self.position_size == 0.0)
        return is_closed, realized_pnl, tid



