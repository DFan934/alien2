# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# prediction_engine/execution/manager.py – feature‑vector dimension aligned
# ---------------------------------------------------------------------------
from __future__ import annotations

"""Execution Manager
====================
Coordinates signal generation → risk sizing → safety gating → order
emission/logging.  Now integrates the **Five‑Tier Safety FSM** and downside‑
variance Kelly sizing described in the BO⭑BBY blueprint.

Place this file at ``prediction_engine/execution/manager.py``.
"""

import asyncio
import contextlib
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from prediction_engine.utils.latency import timeit
from prediction_engine.ev_engine import EVEngine
from .risk_manager import RiskManager
from .safety import SafetyFSM, HaltReason

logger = logging.getLogger(__name__)


class ExecutionManager:  # pylint: disable=too-many-instance-attributes
    """Orchestrates EV evaluation, Kelly sizing, *and* safety checks."""

    def __init__(
        self,
        ev: EVEngine,
        risk_mgr: RiskManager,
        lat_monitor: Any,  # expects .mean attr (ms); keep loose for extensibility
        config: Dict[str, Any],
        log_path: str | Path,
    ) -> None:
        self.ev = ev
        self.risk_mgr = risk_mgr
        self.lat_monitor = lat_monitor
        self.safety = SafetyFSM(config.get("safety", {}))

        self._sig_q: "asyncio.Queue[str]" = asyncio.Queue(maxsize=config.get("max_queue", 5000))
        self._writer_task: Optional[asyncio.Task] = None
        self._log_path = Path(log_path)
        self._numeric_keys: Optional[list[str]] = None

    # ------------------------------------------------------------------
    async def start(self) -> None:
        """Launch background writer."""
        self._writer_task = asyncio.create_task(self._signal_writer())

    async def stop(self) -> None:
        """Flush queue & stop writer."""
        if self._writer_task:
            await self._sig_q.join()
            self._writer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._writer_task

    # ------------------------------------------------------------------
    @timeit("execution_bar")
    async def on_bar(self, bar: Dict[str, Any]) -> None:  # noqa: C901
        """Primary entry called by the bar‑ingestion loop."""
        sym = bar["symbol"]
        price = bar["price"]
        adv = bar["adv"]
        feats: Dict[str, Any] = bar["features"]


        # 0) Live metrics for safety check -----------------------------
        latency_ms: float = float(getattr(self.lat_monitor, "mean", 0.0))
        atr = float(feats.get("atr", 0.0))
        trade_loss = float(getattr(self.risk_mgr, "last_loss", 0.0))
        day_pl = float(getattr(self.risk_mgr, "day_pl", 0.0))

        #drawdown = float(getattr(self.risk_mgr, "drawdown", 0.0))

        dd_obj = getattr(self.risk_mgr, "drawdown", 0.0)
        drawdown = float(dd_obj() if callable(dd_obj) else dd_obj)

        if self.safety.should_halt(
            latency_ms=latency_ms,
            symbol_volatility=atr,
            #volatility=atr or 0.0,
            trade_loss=trade_loss,
            day_pl=day_pl,
            drawdown=drawdown,
        ):
            logger.info("Safety halt active (%s); skipping bar for %s", self.safety.active_reason.name, sym)
            return

        # 1) Update ATR for post‑trade risk calculations --------------
        if atr > 0:
            self.risk_mgr.update_atr(sym, atr)

        # 2) Base position sizing -------------------------------------
        base_qty = self.risk_mgr.desired_size(sym, price)
        if base_qty <= 0:
            return

        # 3) Numeric feature vector -----------------------------------
        if self._numeric_keys is None:
            self._numeric_keys = [k for k, v in feats.items() if isinstance(v, (int, float)) and not math.isnan(v)]
            self._numeric_keys.sort()
            exp_dim = self.ev.centers.shape[1]
            if len(self._numeric_keys) > exp_dim:
                logger.warning("%d numeric cols > EVEngine dim %d; truncating", len(self._numeric_keys), exp_dim)
                self._numeric_keys = self._numeric_keys[:exp_dim]

        exp_dim = self.ev.centers.shape[1]
        if len(self._numeric_keys) != exp_dim:
            logger.warning("Feature dimension %d ≠ EVEngine expectation %d. Skipping bar.", len(self._numeric_keys), exp_dim)
            return

        x_vec = np.array([feats[k] for k in self._numeric_keys], dtype=np.float32)

        #ev_res = self.ev.evaluate(x_vec, base_qty, adv)
        ev_res = self.ev.evaluate(x_vec, adv_percentile=adv)

        # 4) Kelly scaling (downside variance) -------------------------
        #denom = max(ev_res.variance_down, 1e-8)
        denom = self.risk_mgr.scale_variance(ev_res.variance_down, adv)

        kelly_frac = max(0.0, ev_res.mu / (denom * 2))
        final_qty = int(base_qty * min(kelly_frac, 1.0))
        if final_qty <= 0:
            return

        # 5) Signal out -----------------------------------------------
        await self._sig_q.put(
            json.dumps(
                {
                    "ts": time.time(),
                    "sym": sym,
                    "qty": final_qty,
                    "price": price,
                    "exp_ret": ev_res.mu,
                    "var": ev_res.sigma,
                    "cluster": ev_res.cluster_id,
                }
            )
        )

    # ------------------------------------------------------------------
    async def on_fill(self, fill: Dict[str, Any]) -> None:
        """Pass fill info to risk & safety modules."""
        pnl = self.risk_mgr.process_fill(fill)  # returns trade P&L (could be 0 for partial)
        symbol_vol = self.risk_mgr.atr(fill["symbol"])  # type: ignore[arg-type]
        self.safety.register_fill(trade_loss=-pnl, symbol_volatility=symbol_vol)

    # ------------------------------------------------------------------
    @timeit("signal_writer")
    async def _signal_writer(self) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", buffering=1) as fp:
            while True:
                line = await self._sig_q.get()
                fp.write(line + "\n")
                self._sig_q.task_done()
