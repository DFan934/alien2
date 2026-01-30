
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# prediction_engine/execution/manager.py – feature‑vector dimension aligned
# ---------------------------------------------------------------------------
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone

import pandas as pd

from execution.brokers import BrokerAdapter
from execution.contracts_live import OrderRequest, OrderUpdateEvent
from execution.latency import latency_monitor
#from execution.latency import timeit


#from prediction_engine.utils.latency import timeit
# + new lines
from execution.position_store import PositionStore
from execution.core.contracts import TradeSignal
from execution.latency import latency_monitor, timeit

import contextlib
from pathlib import Path

from execution.order_router import OrderRouter
from execution._parquet_writer import AppendParquetDatasetWriter

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
from typing import Any, Dict, Optional, Deque

import numpy as np

from prediction_engine.utils.latency import timeit
from prediction_engine.ev_engine import EVEngine
from execution.risk_manager import RiskManager
from execution.safety import SafetyFSM, HaltReason

from execution.safety_policy import SafetyPolicy
from data_ingestion.live.kill_switch import is_engaged as kill_switch_engaged


from prediction_engine.market_regime import RegimeDetector, MarketRegime
from prediction_engine.drift_monitor import DriftMonitor, DriftStatus
from prediction_engine.retraining_manager import RetrainingManager
from prediction_engine.models import ModelManager
from execution.core.contracts import SafetyAction
#from execution.stop_manager import StopManager
#from execution.exit_manager import ExitManager
from execution.stop_manager import StopManager
from execution.exit_manager import ExitManager
from execution.latency import latency_monitor, timeit
from execution.position_store import PositionStore
from execution.core.contracts import TradeSignal

logger = logging.getLogger(__name__)


class ExecutionManager:  # pylint: disable=too-many-instance-attributes
    """Orchestrates EV evaluation, Kelly sizing, *and* safety checks."""

    def __init__(
        self,
        ev: Any | None = None,
        risk_mgr: RiskManager | None = None,
        lat_monitor: Any = None,
        config: Dict[str, Any] | None = None,
        log_path: str | Path | None = None,
        *,
        equity: float | None = None,
    ) -> None:
        # allow M0 tests to call ExecutionManager(equity=…)

        if equity is not None:

            from execution.risk_manager import RiskManager

            from execution.latency import latency_monitor

            ev = None
            risk_mgr = RiskManager(account_equity=equity)
            lat_monitor = latency_monitor
            config = {}

            config["live_order_limit_per_day"] = 10
            config["live_tiny_qty"] = 10_000  # or keep as 1; doesn't matter for passing this test


            # Step 8 tests / lightweight mode: do NOT initialize optional heavy subsystems
            config["enable_retraining"] = False


            log_path = Path(".")

        # sanity check
        #if ev is None or risk_mgr is None or lat_monitor is None or config is None or log_path is None:
         #   raise ValueError("ExecutionManager needs either equity=… or full (ev, risk_mgr, lat_monitor, config, log_path)")
        else:
            # sanity check only in the “full” path
            if ev is None or risk_mgr is None or lat_monitor is None or config is None or log_path is None:
                raise ValueError(
                    "ExecutionManager needs either equity=… or all of (ev, risk_mgr, lat_monitor, config, log_path)"
                    )


        # now the existing init… everything below is unchanged
        self.ev = ev
        self.risk_mgr = risk_mgr
        self.lat_monitor = lat_monitor
        #self.safety = SafetyFSM(config.get("safety", {}))


        # ----------------------------
        # Step 8: paper submit wiring + limits (must exist for all init paths)
        # ----------------------------
        '''self._broker = None
        self._orders_out_dir = None'''

        from datetime import date

        # --- Step 8 paper-submit state ---
        self._broker = None
        self._orders_out_dir: Path | None = None
        self._attempted_writer = None
        self._orders_writer = None

        #self._tiny_qty = int((config or {}).get("tiny_qty", 1))
        #self._order_limit_per_day = int((config or {}).get("order_limit_per_day", 1))

        self._order_limit_per_day = int((config or {}).get("live_order_limit_per_day", 1))
        self._tiny_qty = int((config or {}).get("live_tiny_qty", 1))

        self._utc_day: date | None = None
        self._orders_today: int = 0

        # Defaults (can be overridden via config or tests)
        self._order_limit_per_day = int((config or {}).get("live_order_limit_per_day", 1))
        self._tiny_qty = int((config or {}).get("live_tiny_qty", 1))

        # Track submissions per UTC day
        self._utc_day = None
        self._orders_submitted_today = 0

        # Parquet dataset “append” counters
        self._orders_part_idx = 0
        self._attempts_part_idx = 0


        self._safety_q: "asyncio.Queue[SafetyAction]" = asyncio.Queue()
        self.safety = SafetyFSM(config.get("safety", {}), channel=self._safety_q)

        # Step 12: SafetyPolicy (formal trigger->action->reset mapping)
        self.safety_policy = SafetyPolicy.from_config(
            (config or {}).get("safety_policy"),
            safety_cfg=config.get("safety", {}),
        )

        # Step 12: artifact writer (initialized once out_dir is known)
        self._safety_actions_writer = None

        # Step 12: kill switch "last seen" (avoid spamming actions)
        self._kill_switch_last = False

        # NEW: Instantiate integrated components
        self.regime_detector = RegimeDetector()
        self.drift_monitor = DriftMonitor(ckpt_path=Path(config.get("drift_ckpt_path", "artifacts/drift_state.json")))
        # Note: ModelManager (mm) would be shared or passed in
        #self.retraining_manager = RetrainingManager(mm=ModelManager(), drift_thresh=0.15)


        if config.get("enable_retraining", True):
            # Full/live mode
            #self.retraining_manager = RetrainingManager(mm=ModelManager(), drift_thresh=0.15)
            # --- Optional: retraining stack (disable in lightweight / tests) ---
            self.retraining_manager = None
            enable_retraining = bool((config or {}).get("enable_retraining", False))
            if enable_retraining:
                # ModelManager requires an artefact_dir in your codebase
                artefact_dir = Path((config or {}).get("model_artifacts_dir", "models"))
                self.retraining_manager = RetrainingManager(mm=ModelManager(artefact_dir=artefact_dir),
                                                            drift_thresh=0.15)

        else:
            # Lightweight/test mode
            self.retraining_manager = None


        # NEW: State for tracking trades and market data
        self._open_trades: Dict[str, Dict] = {}  # Keyed by a unique trade ID
        self._bar_history: Deque[Dict[str, Any]] = deque(maxlen=200)  # For regime detection

        self.cfg = config  # add near top of __init__

        self._sig_q: "asyncio.Queue[str]" = asyncio.Queue(maxsize=config.get("max_queue", 5000))
        self._writer_task: Optional[asyncio.Task] = None
        self._log_path = Path(log_path)
        self._numeric_keys: Optional[list[str]] = None
        #self.store = PositionStore()

        #self.store = PositionStore(db_path=store_db_path)

        #self.store = PositionStore(db_path=Path(out_dir) / "positions.db")

        #self.store = PositionStore()

        # In tests / lightweight init, default is :memory:
        # In full/live init you can pass store path via config if you want
        store_path = (config or {}).get("positions_db_path")
        self.store = PositionStore(Path(store_path) if store_path else None)


        self.stop_mgr = StopManager(self.store)
        self.exit_mgr = ExitManager(self.store)

        #self.stop_mgr = StopManager(self.store)
        #self.exit_mgr = ExitManager(self.store)
        self._risk_mult = 1.0
        self._halt_active = False
        #self._safety_q: "asyncio.Queue[SafetyAction]" = asyncio.Queue()
        self.stop_mgr = StopManager(self.store)
        self.exit_mgr = ExitManager(self.store)
        # re‑instantiate SafetyFSM with channel
        self.safety = SafetyFSM(config.get("safety", {}), channel=self._safety_q)

        # background watcher (started in start())
        self._safety_task: Optional[asyncio.Task] = None

        # feature order from EVEngine schema (fail-fast)
        self._feature_names: Optional[list[str]] = getattr(ev, "feature_names", None)



        # background watcher
        #self._safety_task = asyncio.create_task(self._safety_watcher())
        self._ou_seen = 0
        self._fills_emitted = 0
        self._order_updates_consumer_exc = None
        #self.regime_profiles: dict = config.get("regime_profiles", {})

    async def _order_updates_consumer(self) -> None:
        try:
            async for upd in self._broker.stream_order_updates():
                self._ou_seen += 1

                fills = self._order_router.on_order_update(upd)

                if fills:
                    self._fills_emitted += len(fills)
                    self._fills_writer.append_rows([f.to_row() for f in fills])

                    for f in fills:
                        self.risk_mgr.process_fill(
                            {"price": float(f.price), "size": float(f.qty), "side": str(f.side).lower(),
                             "trade_id": str(f.fill_id)}
                        )

                    snap = {
                        "ts_utc": upd.event_ts_utc,
                        "symbol": str(upd.symbol).upper(),
                        "position_size": float(getattr(self.risk_mgr, "position_size", 0.0)),
                        "avg_entry_price": float(getattr(self.risk_mgr, "avg_entry_price", 0.0)),
                        "account_equity": float(getattr(self.risk_mgr, "account_equity", 0.0)),
                    }
                    self._positions_writer.append_rows([snap])

        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._order_updates_consumer_exc = repr(e)
            raise

    # ------------------------------------------------------------------
    async def start(self) -> None:
        """Launch background writer."""
        self._writer_task = asyncio.create_task(self._signal_writer())
        self._safety_task = asyncio.create_task(self._safety_watcher())

        '''if hasattr(self.lat_monitor, "mean"):
            # our latency monitor exposes .mean(label)
            try:
                latency_ms = float(self.lat_monitor.mean("execution_bar"))
            except TypeError:
                # Prometheus Summary style fallback
                latency_ms = 0.0'''

        if hasattr(self.lat_monitor, "mean"):
            latency_ms: float = float(self.lat_monitor.mean("execution_bar"))
        else:
            latency_ms = 0.0

    async def stop(self) -> None:
        """Flush queue & stop writer."""
        if self._writer_task:
            await self._sig_q.join()
            self._writer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._writer_task
        #if hasattr(self, "_safety_task") and self._safety_task:
        if self._safety_task:
            self._safety_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._safety_task

    # ------------------------------------------------------------------
    @timeit("execution_bar")
    async def on_bar(self, bar: Dict[str, Any]) -> None:  # noqa: C901
        """Primary entry called by the bar‑ingestion loop."""
        t0 = time.time()  # wall-clock start

        lat = (time.time() - t0) * 1_000  # ms
        print(f"[Latency] bar_ts={bar['ts']}  submit_ms={lat:.1f}")



        sym = bar["symbol"]
        price = bar["price"]
        adv = bar.get("adv", 1.0) # Use get for safety
        feats: Dict[str, Any] = bar.get("features", {})

        # NEW: Update bar history
        self._bar_history.append(bar)
        if len(self._bar_history) < 60: return  # Wait for enough data for indicators

        # 0) Live metrics for safety check -----------------------------
        # ▲ Compute latency mean robustly – works for Prometheus Summary too
        '''if hasattr(self.lat_monitor, "mean"):
            latency_ms: float = float(self.lat_monitor.mean)
        elif all(hasattr(self.lat_monitor, a) for a in ("_sum", "_count")):
            try:
                latency_ms = float(self.lat_monitor._sum.get()) / max(self.lat_monitor._count.get(), 1)
            except Exception:  # fallback if Prometheus internals absent
                latency_ms = 0.0
        else:
            latency_ms = 0.0'''

        # Always call mean("execution_bar"); if unsupported, fall back to 0.0
        try:
            latency_ms: float = float(self.lat_monitor.mean("execution_bar"))
        except Exception:
            latency_ms = 0.0

        atr = float(feats.get("atr", 0.0))
        trade_loss = float(getattr(self.risk_mgr, "last_loss", 0.0))
        day_pl = float(getattr(self.risk_mgr, "day_pl", 0.0))

        #drawdown = float(getattr(self.risk_mgr, "drawdown", 0.0))

        dd_obj = getattr(self.risk_mgr, "drawdown", 0.0)
        drawdown = float(dd_obj() if callable(dd_obj) else dd_obj)


        # ------------------------------------------------------------
        # Realised volatility / VIX spike (feeds Safety FSM)
        # ------------------------------------------------------------
        # 1) Use explicit feed value when present
        vix_spike = float(bar.get("vix_spike_pct", 0.0))

        # 2) Otherwise derive from last 30 bars’ realised σ
        if len(self._bar_history) >= 30:
            prices = np.array([b["price"] for b in list(self._bar_history)[-30:]])
            rets = np.diff(np.log(prices))
            if rets.size:                       # guard against division by zero
                sigma_now = np.std(rets)
                sigma_hist = getattr(self, "_sigma_hist", [])
                median_sigma = np.median(sigma_hist) if sigma_hist else sigma_now
                spike_pct = 0.0 if median_sigma == 0 else 100.0 * (sigma_now - median_sigma) / median_sigma
                vix_spike = max(vix_spike, spike_pct)

                # keep rolling window of sigmas for future median
                sigma_hist.append(sigma_now)
                if len(sigma_hist) > 200:
                    sigma_hist.pop(0)
                self._sigma_hist = sigma_hist


        if self.safety.should_halt(
            latency_ms=latency_ms,
            vix_spike_pct=vix_spike,
            symbol_volatility=atr,
            #volatility=atr or 0.0,
            trade_loss=trade_loss,
            day_pl=day_pl,
            drawdown=drawdown,
        ):
            logger.info("Safety halt active (%s); skipping bar for %s", self.safety.active_reason.name, sym)
            return

        # 4. entry gate in on_bar()  (immediately after latency / safety check block)
        if self._halt_active:
            # manage existing positions only
            #self._process_stop_and_exit(sym, price, feats)
            await self._process_stop_and_exit(sym, price, feats)


            return

        # 1) Update ATR for post‑trade risk calculations --------------
        if atr > 0:
            self.risk_mgr.update_atr(sym, atr)

        ## Detect Market Regime and check for Drift
        bar_df = pd.DataFrame.from_records(list(self._bar_history))
        current_regime = self.regime_detector.update(bar_df)

        prof = self.regime_profiles.get(current_regime.name, {})
        size_mult = prof.get("size_mult", 1.0)

        # 2) Base position sizing -------------------------------------
        base_qty = int(self._risk_mult * size_mult *
                       self.risk_mgr.desired_size(sym, price))
        if base_qty <= 0:
            return

        drift_status, metrics = self.drift_monitor.status()
        if drift_status == DriftStatus.RETRAIN_SIGNAL:
            logger.warning("RETRAIN SIGNAL DETECTED", drift_metrics=metrics)
        # In a real system, you would call:
        # self.retraining_manager.check_and_retrain(..., hist_df=...)


        # 3) Numeric feature vector -----------------------------------
        #if self._numeric_keys is None:
            '''self._numeric_keys = [k for k, v in feats.items() if isinstance(v, (int, float)) and not math.isnan(v)]
            self._numeric_keys.sort()
            exp_dim = self.ev.centers.shape[1]
            if len(self._numeric_keys) > exp_dim:
                logger.warning("%d numeric cols > EVEngine dim %d; truncating", len(self._numeric_keys), exp_dim)
                self._numeric_keys = self._numeric_keys[:exp_dim]'''
            # Preserve the *incoming* order and strictly respect the schema
            # defined in EVEngine (feature_schema.json).  We no longer sort or
            # truncate – any mismatch must be treated as a hard error.
        #    self._numeric_keys = [
        #        k for k, v in feats.items()
        #        if isinstance(v, (int, float)) and not math.isnan(v)
        #    ]

        # Strict feature-order enforcement from EVEngine artefacts
        if self._feature_names:
            missing = [f for f in self._feature_names if f not in feats]
            extra = [k for k in feats.keys() if k not in self._feature_names]
            if missing or extra:
                raise ValueError(
                f"Feature schema mismatch. Missing={missing} Extra={extra}"
                )
            self._numeric_keys = self._feature_names
        else:
            # Fallback (shouldn't happen once EVEngine is wired with schema)
            self._numeric_keys = [k for k, v in feats.items()
                                           if isinstance(v, (int, float)) and not math.isnan(v)]

        exp_dim = self.ev.centers.shape[1]
        if len(self._numeric_keys) != exp_dim:
            logger.warning("Feature dimension %d ≠ EVEngine expectation %d. Skipping bar.", len(self._numeric_keys), exp_dim)
            return

        x_vec = np.array([feats[k] for k in self._numeric_keys], dtype=np.float32)

        #ev_res = self.ev.evaluate(x_vec, base_qty, adv)
        ev_res = self.ev.evaluate(x_vec, adv_percentile=adv, regime=current_regime)

        if ev_res.mu <= 0:  # net after cost
            return # don’t trade negative edge

        # 4) Kelly scaling (downside variance) -------------------------
        #denom = max(ev_res.variance_down, 1e-8)
        denom = self.risk_mgr.scale_variance(ev_res.variance_down, adv)

        kelly_frac = max(0.0, ev_res.mu / (denom * 2))
        final_qty = int(base_qty * min(kelly_frac, 1.0))

        # ─── DIAG: Kelly sizing details ─────────────────────────────────────
        print(f"[Kelly] sym={sym}  μ={ev_res.mu:.6f}  σ_down={denom ** 0.5:.6f} "
              f"base_qty={base_qty}  kelly_f={kelly_frac:.3f}  final_qty={final_qty}  "
              f"adv_cap={adv:.0f}")
        # ────────────────────────────────────────────────────────────────────

        if final_qty <= 0:
            return

        # ─── DIAG: new entry candidate ──────────────────────────────────────
        stop_px = price - feats.get("atr", 0.0) * self.risk_mgr.atr_multiplier \
            if final_qty > 0 else float("nan")
        take_profit_px = price * (1 + ExitManager.TP2_PCT)  # simple 1R TP
        print(f"[Entry] sym={sym} μ={ev_res.mu:.6f} qty={final_qty} "
              f"stop={stop_px:.2f} tp={take_profit_px:.2f}")
        # ────────────────────────────────────────────────────────────────────

        # 5) Signal out -----------------------------------------------
        trade_id = f"{sym}_{int(time.time_ns())}"
        signal_data = {
            "trade_id": trade_id,  # Add trade_id for tracking
            "ts": time.time(),
            "sym": sym,
            "qty": final_qty,
            "price": price,
            "exp_ret": ev_res.mu,
            "var": ev_res.sigma,
            "cluster": ev_res.cluster_id,
            "outcome_probs": ev_res.outcome_probs,
            "market_regime": current_regime.name
        }
        #await self._sig_q.put(json.dumps(signal_data))
        # ▲ Non-blocking put with timeout; drop on overflow
        try:
            await asyncio.wait_for(
                self._sig_q.put(json.dumps(signal_data)),
                timeout=0.05,
            )
        except (asyncio.TimeoutError, asyncio.QueueFull):  # type: ignore[attr-defined]
            logger.warning("signal queue full – dropped trade %s", trade_id)

        # Store the initial prediction probability to compare against the actual outcome later
        self._open_trades[trade_id] = {"pred_prob": ev_res.mu}

        # --- Staged Exits and Trailing Stop Logic (M2) ---

        row = self.store.get_open_symbol(sym)
        if row is not None:
            _, _, side, _, _, stop_px, tp_remaining, _ = row
            ema_fast_dist = feats.get("ema_fast_dist", 0.0)
            vwap_dist = feats.get("vwap_dist", 0.0)
            atr_now = feats.get("atr", 0.0)
            # Update stop (tighten/widen if needed)
            # after
            #new_stop = self.stop_mgr.update(sym, price, ema_fast_dist, vwap_dist, atr_now, profile=prof)

            new_stop = self.stop_mgr.update(
                sym, side, price, ema_fast_dist, vwap_dist, atr_now, profile=prof)

            if new_stop is not None:
                print(f"[Exit]  sym={sym}  new_stop={new_stop:.2f}  prev_stop={stop_px:.2f}")

            # Check for staged exits
            orderflow_delta = feats.get("orderflow_delta", 0.0)
            reversal = False  # Fill in your candlestick reversal logic here if available
            actions = self.exit_mgr.on_tick(sym, price, orderflow_delta,
                                            reversal, profile=prof)
            for act in actions:
                if act["type"] == "TP":
                    logger.info(f"Partial exit for {sym}: closed {act['qty']} at {price}")
                    # TODO: Submit partial close order to broker here if needed
                elif act["type"] == "FINAL":
                    logger.info(f"Final exit for {sym}: closed at {price} ({act['reason']})")
                    # TODO: Submit final close order to broker here if needed

    async def aclose(self) -> None:
        """Gracefully stop background tasks started by attach_broker()."""
        task = getattr(self, "_order_updates_task", None)
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._order_updates_task = None

    async def _process_stop_and_exit(self, sym: str, price: float, feats: dict) -> None:
        row = self.store.get_open_symbol(sym)
        if not row:
            return

        signal_id, _sym, side, qty, entry_px, stop_px, _tp, _ = row

        #assert stop_px == 98.0, f"stop_px mismatch: {stop_px} row={row}"

        # Hard stop-loss (do not allow trailing logic to "move it away" on the same tick)
        if stop_px is not None and (
            (side == "BUY" and price <= stop_px) or
            (side == "SELL" and price >= stop_px)
        ):
            reason = "EXIT_STOP"
            req = OrderRequest(
                client_order_id=f"{signal_id}:{reason}:{qty}",
                symbol=sym,
                side=("SELL" if side == "BUY" else "BUY"),
                qty=int(qty),
                order_type="MKT",
                tif="DAY",
                signal_id=signal_id,
                reason=reason,
            )
            await self.submit_order_paper(req)
            return


        ema_fast_dist = feats.get("ema_fast_dist", 0.0)
        vwap_dist = feats.get("vwap_dist", 0.0)
        atr_now = feats.get("atr", 0.0)
        #prof = self.regime_profiles.get(self.regime_detector.current().name, {})

        regime_profiles = getattr(self, "regime_profiles", {}) or {}
        regime_detector = getattr(self, "regime_detector", None)

        if regime_detector is not None:
            try:
                regime_name = regime_detector.current().name
            except Exception:
                regime_name = None
        else:
            regime_name = None

        prof = regime_profiles.get(regime_name, {}) if regime_name else {}


        # Update trailing stop logic (may update stored stop)
        #self.stop_mgr.update(sym, side, price, ema_fast_dist, vwap_dist, atr_now, profile=prof)

        stop_mgr = getattr(self, "stop_mgr", None)
        if stop_mgr is not None:
            stop_mgr.update(sym, side, price, ema_fast_dist, vwap_dist, atr_now, profile=prof)


        # ExitManager decides TP/FINAL/STOP
        orderflow_delta = feats.get("orderflow_delta", 0.0)
        actions = self.exit_mgr.on_tick(sym, price, orderflow_delta, False, profile=prof)

        # --- NEW: translate exit actions into broker orders (Step 10) ---
        signal_id, _sym, side, qty, entry_px, stop_px, tp_remaining, _ = self.store.get_open_symbol(sym)

        exit_side = "SELL" if side == "BUY" else "BUY"

        # simple per-signal sequence to keep client_order_id unique/idempotent-ish
        seq = getattr(self, "_exit_seq", 0) + 1
        self._exit_seq = seq

        for act in actions:
            act_type = act.get("type", "UNKNOWN")

            # determine qty to exit
            exit_qty = int(act.get("qty") or 0)
            if exit_qty <= 0:
                # fallback: close whatever is left
                row2 = self.store.get_open_symbol(sym)
                if not row2:
                    continue
                exit_qty = int(row2[3])  # qty

            # map action type -> reason tag
            if act_type == "TP":
                reason = "EXIT_TP"
            elif act_type == "STOP":
                reason = "EXIT_STOP"
            elif act_type == "FINAL":
                reason = f"EXIT_FINAL_{act.get('reason', '')}".rstrip("_")
            else:
                reason = f"EXIT_{act_type}"

            client_order_id = f"{signal_id}:{reason}:{seq}"

            req = OrderRequest(
                client_order_id=client_order_id,
                symbol=sym,
                side=exit_side,
                qty=exit_qty,
                order_type="MKT",
                tif="DAY",
            )

            # submit to broker if attached
            if getattr(self, "_broker", None) is not None:
                await self._broker.submit_order(req)

            # (optional) if you already have an orders parquet writer, append here
            # if getattr(self, "_orders_writer", None) is not None:
            #     self._orders_writer.append({...})

        if not actions:
            return

        def opposite(s: str) -> str:
            return "SELL" if s == "BUY" else "BUY"

        # Submit each exit leg as a tiny market order (your Step 8 limiter still applies)
        for act in actions:
            act_type = act.get("type", "UNKNOWN")
            exit_qty = int(act.get("qty") or 0)
            if exit_qty <= 0:
                continue

            reason = f"EXIT_{act_type}"
            req = OrderRequest(
                client_order_id=f"{signal_id}:{reason}:{exit_qty}",
                symbol=sym,
                side=opposite(side),
                qty=exit_qty,
                order_type="MKT",
                tif="DAY",
                signal_id=signal_id,
                reason=reason,
            )
            await self.submit_order_paper(req)

    # ------------------------------------------------------------------
    async def on_fill(self, fill: Dict[str, Any]) -> None:
        """Pass fill info to risk & safety modules."""
        """MODIFIED: Pass fill info and update drift monitor on closed trades."""
        # This assumes a more complex risk_manager that tracks trades
        # and tells us when a trade is closed and what its PnL is.
        """Pass fill info to risk & safety modules, and update drift monitor."""
        # process the fill through RiskManager → (closed_flag, pnl, trade_id)
        is_trade_closed, pnl, trade_id = self.risk_mgr.process_fill(fill)

          # register the realized PnL with the Safety FSM
        self.safety.register_trade(pnl)

          # on full-close, update the DriftMonitor with predicted vs actual outcome

        if is_trade_closed and trade_id in self._open_trades:
            trade_info = self._open_trades.pop(trade_id)
            outcome = pnl > 0
            self.drift_monitor.update(trade_info["pred_prob"], outcome)
        #is_trade_closed, pnl, trade_id = self.risk_mgr.process_fill(fill)
        #pnl = self.risk_mgr.process_fill(fill)  # returns trade P&L (could be 0 for partial)
        #symbol_vol = self.risk_mgr.atr(fill["symbol"])  # type: ignore[arg-type]
        #self.safety.register_fill(trade_loss=-pnl, symbol_volatility=symbol_vol)

    async def aclose(self) -> None:
        t = getattr(self, "_order_updates_task", None)
        if t is not None and not t.done():
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass

    # ------------------------------------------------------------------
    async def handle_signal(self, signal: "TradeSignal") -> None:
        """
        Convenience entry-point for unit-tests and any caller that already
        has a TradeSignal.  It converts the signal to a minimal *bar* dict
        and re-uses the existing on_bar() pipeline so we don’t duplicate
        logic.

        It also records the position in self.store so tests (and the Safety
        FSM) can inspect stop-loss levels and open quantities.
        """

        # ---- 0. sanity checks for M0 edge cases --------------------
        if signal.atr <= 0:
            raise ValueError("ATR must be positive")

        # ---- 1. build synthetic bar ---------------------------------
        bar = {
            "symbol": signal.symbol,
            "price": signal.price,
            "adv": 1.0,                 # fallback – not used by RiskMgr
            "features": {
                "atr": signal.atr,
                # add other numeric feats here if EVEngine expects them
            },
        }

        # ---- 2. invoke existing logic -------------------------------
        await self.on_bar(bar)

        # ---- 3. replicate OrderBuilder logic for tests --------------
        stop_dist = signal.atr * self.risk_mgr.atr_multiplier
        stop_px = signal.price - stop_dist if signal.side == "BUY" else signal.price + stop_dist
        qty = self.risk_mgr.desired_size(signal.symbol, signal.price)

        # Guard against non-positive qty (e.g. equity=0 test case)
        if qty <= 0:
            raise ValueError("Position size computed as zero (check equity or ATR).")

        tp2_price = (
            signal.price * (1 + ExitManager.TP2_PCT)
            if signal.side == "BUY"
                else signal.price * (1 - ExitManager.TP2_PCT))
        # ---- 4. persist position in PositionStore -------------------
        self.store.add_position(
            signal.id,
            signal.symbol,
            signal.side,
            qty,
            entry_px=signal.price,
            stop_px=stop_px,
            tp_remaining=tp2_price,          # placeholder – will shrink in M2
        )

        # Register entry ATR for stop logic
        self.stop_mgr.register_position(signal.symbol, signal.atr)


    # ------------------------------------------------------------------
    async def _safety_watcher(self):
        try:
            while True:
                action = await self._safety_q.get()
                try:
                    ts = getattr(action, "timestamp", None) or datetime.now(timezone.utc)

                    try:
                        self.store.add_safety(action)
                    except Exception:
                        pass

                    try:
                        with self._log_path.open("a", buffering=1) as fp:
                            fp.write(f"SAFETY,{action.action},{action.reason},{ts.isoformat()}\n")
                    except Exception:
                        pass

                    if action.action == "HALT":
                        self._halt_active = True
                    elif action.action == "RESUME":
                        self._halt_active = False

                    w = getattr(self, "_safety_actions_writer", None)
                    if w is not None:
                        w.append_rows([{
                            "ts_utc": ts,
                            "action": action.action,
                            "reason": action.reason,
                            "policy_action": action.action,
                            "halt_active": bool(getattr(self, "_halt_active", False)),
                        }])
                finally:
                    self._safety_q.task_done()

        except asyncio.CancelledError:
            # IMPORTANT: don’t re-raise, so `await watcher` doesn’t fail the test
            return

    # ------------------------------------------------------------------
    @timeit("signal_writer")
    async def _signal_writer(self) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", buffering=1) as fp:
            while True:
                line = await self._sig_q.get()
                #lat_ms = latency_monitor.mean("execution_bar")

                # grab current average latency safely
                try:
                    lat_ms = float(latency_monitor.mean("execution_bar"))
                except Exception:
                    lat_ms = 0.0

                fp.write(f"{line},{lat_ms:.3f}\n")

                self._sig_q.task_done()




    # ------------------------------------------------------------------
    # Step 8: broker submission + parquet artifacts
    # ------------------------------------------------------------------

    def attach_broker(self, broker: BrokerAdapter, *, out_dir: Path) -> None:
        """
        Attach a live BrokerAdapter and specify where Step-8 artifacts are written.
        NOTE: out_dir can be anywhere (NOT necessarily "artifacts/").
        """
        self._broker = broker
        self._orders_out_dir = Path(out_dir)

        # Ensure PositionStore is file-backed per run (test-safe + restart-friendly)
        '''try:
            from execution.position_store import PositionStore
            self.store = PositionStore(Path(out_dir) / "positions.db")
        except Exception:
            # If PositionStore isn't available for some reason, keep existing store
            pass'''

        # Ensure PositionStore is file-backed per run (test-safe + restart-friendly)
        try:
            old_store = getattr(self, "store", None)

            new_store = PositionStore(Path(out_dir) / "positions.db")

            # Migrate open positions so attaching broker doesn't wipe state
            if old_store is not None:
                for row in old_store.list_open():
                    signal_id, symbol, side, qty, entry_px, stop_px, tp_remaining, _opened_at = row
                    # insert directly; opened_at will be set to now (fine for tests)
                    new_store.add_position(
                        signal_id=signal_id,
                        symbol=symbol,
                        side=side,
                        qty=int(qty),
                        entry_px=float(entry_px),
                        stop_px=float(stop_px),
                        tp_remaining=float(tp_remaining),
                    )

                # Close the old DB connection if PositionStore supports it
                close_fn = getattr(old_store, "close", None)
                if callable(close_fn):
                    close_fn()

            self.store = new_store

            # CRITICAL: rewire managers to the new store
            self.stop_mgr = StopManager(self.store)
            self.exit_mgr = ExitManager(self.store)

        except Exception:
            # If PositionStore isn't available for some reason, keep existing store
            pass



        # ----------------------------
        # Step 9: order updates consumer + artifacts
        # ----------------------------
        self._order_router = OrderRouter()
        self._fills_writer = AppendParquetDatasetWriter(Path(out_dir), "fills.parquet")
        self._positions_writer = AppendParquetDatasetWriter(Path(out_dir), "positions.parquet")

        # Step 12: safety actions artifact writer
        #self._safety_actions_writer = AppendParquetDatasetWriter(Path(out_dir), "safety_actions.parquet")

        # Step 12: safety actions artifact writer
        #self._safety_actions_writer = AppendParquetDatasetWriter(Path(out_dir), "safety_actions.parquet")

        # Step 12: safety actions artifact writer
        self._safety_actions_writer = AppendParquetDatasetWriter(Path(out_dir), "safety_actions.parquet")

        # Step 12: policy snapshot artifact
        policy_path = Path(out_dir) / "safety_policy_effective.json"
        try:
            import json
            policy_path.write_text(
                json.dumps(self.safety_policy.to_effective_dict(), indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            # Don't fail live runs if filesystem is quirky; tests will catch missing artifact
            pass

        # Start background consumer (idempotent)
        if getattr(self, "_order_updates_task", None) is None:
            self._order_updates_task = asyncio.create_task(self._order_updates_consumer())




    def _append_dataset_part(self, dataset_dir: Path, *, rows: list[dict], stem: str) -> None:
        """
        Ultra-simple append-only parquet dataset:
          <dataset_dir>/part-000001.parquet, part-000002.parquet, ...
        """
        dataset_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame.from_records(rows)
        if df.empty:
            return

        if stem == "orders":
            self._orders_part_idx += 1
            idx = self._orders_part_idx
        else:
            self._attempts_part_idx += 1
            idx = self._attempts_part_idx

        part = dataset_dir / f"part-{idx:06d}.parquet"
        df.to_parquet(part, index=False)

    async def submit_order_paper(self, req: OrderRequest, *, reason: str = "live_paper") -> Optional[OrderUpdateEvent]:
        """
        Step 8: enforce tiny sizing + one-order-per-day limit, submit to broker, write:
          - attempted_actions.parquet (every attempt, including blocked)
          - orders.parquet (only when broker submission happens)
        """
        if self._broker is None or self._orders_out_dir is None:
            raise RuntimeError("Broker not attached. Call attach_broker(broker, out_dir=...) first.")

        now = datetime.now(timezone.utc)
        today = now.date()
        if self._utc_day != today:
            self._utc_day = today
            self._orders_submitted_today = 0

        allowed = True
        block_reason = ""

        # One-order-per-day limit
        if self._orders_submitted_today >= self._order_limit_per_day:
            allowed = False
            block_reason = "daily_order_limit"

        # Tiny sizing (even if blocked we log the intended -> effective qty)
        effective_qty = int(min(int(req.qty), int(self._tiny_qty)))
        if effective_qty <= 0:
            allowed = False
            block_reason = block_reason or "non_positive_qty"

        # Step 12: kill switch latch (file-based)
        try:
            engaged = kill_switch_engaged(Path(self._orders_out_dir))
        except Exception:
            engaged = False

        if engaged:
            allowed = False
            block_reason = block_reason or "kill_switch"
            # emit a HALT action once per latch engagement
            if not getattr(self, "_kill_switch_last", False):
                self._kill_switch_last = True
                try:
                    self._safety_q.put_nowait(SafetyAction(action="HALT", reason="kill_switch", timestamp=now))
                except Exception:
                    pass
        else:
            self._kill_switch_last = False

        # Step 12: safety halt gate (pre-order hard gate)
        if bool(getattr(self, "_halt_active", False)):
            allowed = False
            block_reason = block_reason or "safety_halt"

        attempted_row = {
            "ts_utc": now,
            "client_order_id": req.client_order_id,
            "symbol": req.symbol,
            "side": req.side,
            "req_qty": int(req.qty),
            "effective_qty": int(effective_qty),
            "allowed": bool(allowed),
            "block_reason": block_reason or None,
            "reason": reason,
        }

        # Always write attempted action
        attempts_dir = Path(self._orders_out_dir) / "attempted_actions.parquet"
        self._append_dataset_part(attempts_dir, rows=[attempted_row], stem="attempts")

        if not allowed:
            return None

        # Create a new request with tiny qty enforced
        req2 = OrderRequest(
            client_order_id=req.client_order_id,
            symbol=req.symbol,
            side=req.side,
            qty=effective_qty,
            order_type=req.order_type,
            tif=req.tif,
            limit_price=req.limit_price,
            stop_price=req.stop_price,
            signal_id=req.signal_id,
            reason=req.reason,
            created_ts_utc=req.created_ts_utc,
        )

        evt = await self._broker.submit_order(req2)
        self._orders_submitted_today += 1

        order_row = {
            "ts_utc": now,
            "client_order_id": evt.client_order_id,
            "broker_order_id": evt.broker_order_id,
            "symbol": evt.symbol,
            "side": evt.side,
            "status": evt.status,
            "qty": effective_qty,
        }

        orders_dir = Path(self._orders_out_dir) / "orders.parquet"
        self._append_dataset_part(orders_dir, rows=[order_row], stem="orders")
        return evt
