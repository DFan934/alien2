# =============================================================================
# prediction_engine/testing_validation/backtester.py  (async‑compatible)
# =============================================================================
"""Async‑compatible back‑tester
================================
Runs a full strategy built around the *async* ExecutionManager loop over a
historical bar DataFrame (or Parquet path).

The strategy object must expose the trio of coroutines:

* ``start()`` – initialise resources (e.g., start signal writer).
* ``on_bar(bar: dict)`` – called for every bar dict with at least
  ``symbol``, ``price``, ``adv``, ``features``.
* ``stop()`` – flush & teardown resources.

Execution latency is captured via the Prometheus metrics exported by
``prediction_engine.monitoring.metrics``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Dict, Tuple

import numpy as np
import pandas as pd
from prometheus_client import REGISTRY

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Broker stub – immediate fills at next‑bar open
# -----------------------------------------------------------------------------
class BrokerStub:
    """Very simple in‑memory broker that fills every submitted order at the
    *next* bar's open price (zero slippage by default but hooks available).
    """

    def __init__(self, slippage_bp: float = 0.0):
        self.slippage_bp = slippage_bp  # basis‑points slippage per trade
        self._pending: List[Tuple[str, int]] = []  # (symbol, qty) until next bar
        self.trades: List[Dict[str, Any]] = []
        self.positions: Dict[str, int] = {}

    def queue_order(self, symbol: str, qty: int):
        self._pending.append((symbol, qty))

    def execute_pending(self, bar: Dict[str, Any]):
        """Executes all queued orders at *this* bar's open price."""
        open_px = bar["price"]
        fills = []
        for sym, qty in self._pending:
            fill_px = open_px * (1 + math.copysign(self.slippage_bp / 1e4, qty))
            self.positions[sym] = self.positions.get(sym, 0) + qty
            trade = {
                "ts": bar["ts"],
                "symbol": sym,
                "qty": qty,
                "price": fill_px,
            }
            self.trades.append(trade)
            fills.append(trade)
        self._pending.clear()
        return fills

    def mark_to_market(self, bar: Dict[str, Any]):
        """Returns unrealised P/L given current bar price."""
        sym = bar["symbol"]
        pos = self.positions.get(sym, 0)
        if pos == 0:
            return 0.0
        last_fill = next((t for t in reversed(self.trades) if t["symbol"] == sym), None)
        if not last_fill:
            return 0.0
        return (bar["price"] - last_fill["price"]) * pos


# -----------------------------------------------------------------------------
# Async Backtester
# -----------------------------------------------------------------------------
class AsyncBacktester:
    def __init__(
        self,
        strategy,  # object with start/on_bar/stop
        bars: pd.DataFrame,
        broker: BrokerStub,
        equity0: float = 100_000,
    ):
        self.strategy = strategy
        self.bars = bars.sort_values("timestamp")
        self.broker = broker
        self.equity0 = equity0
        self.equity_curve: List[Tuple[float, float]] = []  # (ts, equity)

    # ------------------------------------------------------------------
    async def run(self):
        await self.strategy.start()
        cash = self.equity0

        for row in self.bars.itertuples():
            # ------------------------------------------------------------------ price pick
            # ------------------------------------------------------------------ price pick
            price = None

            # 1) explicit column hints, in priority order
            for col in ("open", "Open", "close", "Close", "last", "Last"):
                if hasattr(row, col):
                    price = getattr(row, col)
                    if isinstance(price, (int, float)) and not pd.isna(price):
                        break

            # 2) fallback – first numeric value in the row dict
            if price is None or pd.isna(price):
                for val in row._asdict().values():
                    if isinstance(val, (int, float)) and not pd.isna(val):
                        price = val
                        break

            if price is None or pd.isna(price):
                raise RuntimeError("No numeric price found in this row")
            # ---------------------------------------------------------------------------

            # -----------------------------------------------------------------------------

            # ✂     around line 110 – replace the whole block that builds `bar`
            # -----------------------------------------------------------------
            bar = {
                "ts": row.timestamp,
                "symbol": row.symbol,
                "price": price,  # ← use the price we found
                "adv": getattr(row, "adv", 1_000_000),
                "features": row._asdict(),
            }
            # -----------------------------------------------------------------

            # 1) fill pending orders at this bar open
            fills = self.broker.execute_pending(bar)
            for f in fills:
                cash -= f["qty"] * f["price"]
            # 2) pass bar to strategy
            await self.strategy.on_bar(bar)
            # 3) unrealised P/L on open positions
            mtm = self.broker.mark_to_market(bar)
            equity = cash + mtm
            self.equity_curve.append((bar["ts"], equity))
        await self.strategy.stop()
        return self._summarise()

    # ------------------------------------------------------------------
    def _summarise(self) -> Dict[str, Any]:
        arr = np.array([e for _, e in self.equity_curve])
        rets = np.diff(arr) / arr[:-1]
        sharpe = np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(252)
        sortino = (np.mean(rets) / (np.std(rets[rets < 0]) + 1e-8)) * np.sqrt(252)
        dd = 1.0 - arr / np.maximum.accumulate(arr)
        max_dd = np.max(dd)

        # latency metrics (if any Prometheus Summary named execution_bar_latency_seconds)
        lat_sum = REGISTRY.get_sample_value(
            "execution_bar_latency_seconds_sum") or 0.0
        lat_cnt = REGISTRY.get_sample_value(
            "execution_bar_latency_seconds_count") or 1.0
        avg_latency = lat_sum / lat_cnt

        return {
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "avg_latency_s": avg_latency,
            "final_equity": arr[-1],
        }


# -----------------------------------------------------------------------------
# CLI helper for quick runs  –  `python -m testing_validation.backtester ...`
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from prediction_engine.ev_engine import EVEngine
    from prediction_engine.distance_calculator import DistanceCalculator  # noqa
    from prediction_engine.execution.manager import ExecutionManager
    from prediction_engine.execution.risk_manager import RiskManager

    parser = argparse.ArgumentParser(description="Async back‑test runner")
    parser.add_argument("--parquet", required=True, help="Path to feature parquet slice")
    parser.add_argument("--stats", required=True, help="cluster_stats.npz")
    parser.add_argument("--kernel", required=True, help="kernel_bandwidth.json")
    parser.add_argument("--rf", required=True, help="rf_feature_weights.pkl")
    parser.add_argument("--equity", type=float, default=100_000)
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet)

    ev = EVEngine.from_artifacts(args.stats, args.kernel, args.rf)
    risk = RiskManager(equity=args.equity, max_notional=args.equity * 0.25)
    exec_mgr = ExecutionManager(ev, risk, Path("logs/sim_signals.jsonl"))

    broker = BrokerStub(slippage_bp=2.0)

    bt = AsyncBacktester(exec_mgr, df, broker, equity0=args.equity)
    summary = asyncio.run(bt.run())

    print(json.dumps(summary, indent=2))
