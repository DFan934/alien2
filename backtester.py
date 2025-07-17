#==================================
# backtester.py
#====================================


# backtester.py (Corrected Version)

from __future__ import annotations
import logging
import math
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

import pandas as pd
from prediction_engine.ev_engine import EVEngine
from execution.risk_manager import RiskManager
from prediction_engine.testing_validation.backtester import BrokerStub
from execution.manager import ExecutionManager
# Backtester now receives ready‑made components; only type hints needed
from prediction_engine.ev_engine import EVEngine
from execution.risk_manager import RiskManager
from prediction_engine.tx_cost import BasicCostModel
from typing import Any

class Backtester:
    """
    Runs an event-driven backtest loop using pre-computed features.
    """

    def __init__(self,
                 features_df: pd.DataFrame,
                 cfg: Dict[str, Any],
                 *,
                 ev: EVEngine,
                 risk: RiskManager,
                 broker: Any,
                 cost_model: BasicCostModel,
                 good_clusters: set[int],
                 ):
        """
        Initializes the Backtester with a DataFrame that already contains
        all necessary features.
        """
        self.feats_df = features_df  # Use the features DataFrame directly
        self.cfg = cfg
        self.log = logging.getLogger("Backtester")
        self.pc_cols = [c for c in self.feats_df.columns if c.startswith("pca_")]
        self.ev = ev
        self.risk = risk
        self.broker = broker
        self.cost = cost_model
        self.good_clusters = good_clusters


    def run(self) -> pd.DataFrame:
        """
        Executes the main backtesting event loop.
        """
        # --- 1. Setup ---
        cfg = self.cfg
        #art_dir = Path(cfg["artefacts"])
        #ev = EVEngine.from_artifacts(art_dir)
        ev = self.ev
        #risk = RiskManager(account_equity=float(cfg["equity"]))
        risk = self.risk
        #broker = BrokerStub(slippage_bp=float(cfg.get("slippage_bp", 0.0)))
        broker = self.broker
        equity_curve: List[Dict[str, Any]] = []
        log = self.log

        log.info("Starting backtest with %d bars...", len(self.feats_df))

        # --- 2. Event Loop ---
        for _, row in self.feats_df.iterrows():
            bar = {
                "ts": row["timestamp"],
                "symbol": row["symbol"],
                "price": row["open"],  # Fills at NEXT bar's open
            }

            # 5.1 Execute any queued orders (from previous bar)
            # 5.1 Execute any queued orders (from previous bar)
            fills = broker.execute_pending(bar)
            realised_pnl = 0.0
            # in Backtester.run(), around line 110 or so, in your fills loop:

            for tr in fills:
                fill_dict = {
                    "price": tr["price"],
                    #"size": tr["qty"],
                    "size": abs(tr["qty"]),
                    #"side": "BUY" if tr["qty"] > 0 else "SELL",
                    "side": "BUY" if tr["qty"] > 0 else "SELL",
                    "trade_id": tr.get("trade_id", ""),
                }

                try:
                    is_closed, pnl, tid = risk.process_fill(fill_dict)
                except Exception as e:
                    # something went wrong inside process_fill itself
                    logging.error("process_fill raised %s for fill %r on bar %s",
                                  e, fill_dict, bar["ts"])
                    raise

                # right here—catch any non‑finite P&L or equity
                # right after you compute pnl via process_fill(...)
                if not math.isfinite(pnl):
                    # include the bad pnl, the fill, and the timestamp in the exception
                    raise RuntimeError(
                        f"Bad realized_pnl: {pnl!r} from fill {fill_dict!r} on bar {bar['ts']}"
                    )

                # then your equity check
                if not math.isfinite(risk.account_equity):
                    raise RuntimeError(
                        f"Post‑fill account_equity non‑finite: {risk.account_equity!r} "
                        f"  (pnl={pnl!r}, fill={fill_dict!r}, ts={bar['ts']})"
                    )

                realised_pnl += pnl

            # 5.2 Mark-to-market unrealised P/L
            mtm = broker.mark_to_market(bar)

            # 5.3 Generate signal for NEXT bar
            x_vec = row[self.pc_cols].to_numpy(dtype=np.float32)
            ev_res = ev.evaluate(x_vec, adv_percentile=10.0, regime=None, half_spread=0)
            cid = ev_res.cluster_id
            adv_today = row.get("adv", None)  # avg daily volume if present

            # Only size up if this cluster historically had μ>0:
            if cid in self.good_clusters and ev_res.mu > 0:
                qty = self.risk.kelly_position(
                    mu=ev_res.mu,
                    variance_down=ev_res.variance_down,
                    price=row["open"],
                    adv=adv_today,
                )
            else:
                qty = 0
            # Simplified sizing for this example
            #qty = risk.desired_size(cfg["symbol"], row["open"])

            #if qty != 0:
            #    broker.queue_order(cfg["symbol"], qty)

            # Use the EVEngine’s Kelly‐fraction to size the trade:
            # Allocate position_size × (current equity) in dollars,
            # then convert to a share quantity at next‐bar open.
            #current_cap = risk.account_equity + mtm
            #dollar_allocation = ev_res.position_size * current_cap
            #qty = int(dollar_allocation // row["open"])

            # --- Kelly sizing directly from RiskManager -------------------

            # in Backtester.run, right before qty = risk.kelly_position(...)
            if not math.isfinite(risk.account_equity):
                logging.error("Non‐finite account_equity at bar %s: %s", bar["ts"], risk.account_equity)
                raise RuntimeError("account_equity has become non‐finite")

            '''qty = risk.kelly_position(
                mu = ev_res.mu,
                variance_down = ev_res.variance_down,
                price = row["open"],
                adv = adv_today,
            )'''

            # ─── guard against non‐finite or negative sizes ───────────────────
            #import math, logging
            if not math.isfinite(qty) or qty < 0:
                logging.error(
                    "kelly_position returned bad qty=%r on bar %s (mu=%r, var=%r, price=%r)",
                    qty, bar["ts"], ev_res.mu, ev_res.variance_down, row["open"]
                )
                qty = 0


            #  ⬇ NEW: close any open position after 1‑bar hold
            #qty_exit = 0
            #if risk.position_size > 0:
            #    qty_exit = -risk.position_size  # flatten

            # ─── size exit (1‑bar hold) ───────────────────────────────────────
            if risk.position_size > 0:
                qty_exit = -risk.position_size
                if not math.isfinite(qty_exit):
                    logging.error(
                    "Bad qty_exit=%r on bar %s (position_size=%r)",
                    qty_exit, bar["ts"], risk.position_size
                    )
                    qty_exit = 0
            else:
                qty_exit = 0

            # queue orders for next bar --------------------------------------------
            # queue orders for next bar --------------------------------------------
            if qty_exit != 0:
                # qty_exit is negative, so the broker knows it's a sell
                broker.queue_order(cfg["symbol"], qty_exit)
            elif qty > 0:
                # qty is positive, so the broker knows it's a buy
                broker.queue_order(cfg["symbol"], qty)

            #if qty > 0:
            #    broker.queue_order(cfg["symbol"], qty)

            # --- 3. Record Equity ---
            current_equity = risk.account_equity + mtm
            equity_curve.append({
                "timestamp": bar["ts"],
                "equity": current_equity,
                "realised_pnl": realised_pnl,
                "unrealised_pnl": mtm,
                "qty_next_bar": qty,
                "mu": ev_res.mu,
                "sigma_sq": ev_res.sigma,
                "var_down": ev_res.variance_down,  # NEW
                "residual": ev_res.residual,
                "position_size": ev_res.position_size,
                "open": row["open"],
                "close": row["close"],
                "cluster_id": ev_res.cluster_id,
                "notional_abs": abs(risk.position_size * row["open"]),  #New
            })

        log.info("Backtest loop finished.")
        return pd.DataFrame(equity_curve)