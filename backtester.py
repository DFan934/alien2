#==================================
# backtester.py
#====================================


# backtester.py (Corrected Version)

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

import pandas as pd
from prediction_engine.ev_engine import EVEngine
from execution.risk_manager import RiskManager
from prediction_engine.testing_validation.backtester import BrokerStub


class Backtester:
    """
    Runs an event-driven backtest loop using pre-computed features.
    """

    def __init__(self, features_df: pd.DataFrame, cfg: Dict[str, Any]):
        """
        Initializes the Backtester with a DataFrame that already contains
        all necessary features.
        """
        self.feats_df = features_df  # Use the features DataFrame directly
        self.cfg = cfg
        self.log = logging.getLogger("Backtester")
        self.pc_cols = [c for c in self.feats_df.columns if c.startswith("pca_")]

    def run(self) -> pd.DataFrame:
        """
        Executes the main backtesting event loop.
        """
        # --- 1. Setup ---
        cfg = self.cfg
        art_dir = Path(cfg["artefacts"])
        ev = EVEngine.from_artifacts(art_dir)
        risk = RiskManager(account_equity=float(cfg["equity"]))
        broker = BrokerStub(slippage_bp=float(cfg.get("slippage_bp", 0.0)))

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
            fills = broker.execute_pending(bar)
            realised_pnl = 0.0
            for tr in fills:
                pnl = tr.get("pnl", 0.0)
                risk.on_closed_trade(pnl)
                realised_pnl += pnl

            # 5.2 Mark-to-market unrealised P/L
            mtm = broker.mark_to_market(bar)

            # 5.3 Generate signal for NEXT bar
            x_vec = row[self.pc_cols].to_numpy(dtype=np.float32)
            ev_res = ev.evaluate(x_vec, adv_percentile=10.0, regime=None, half_spread=None)

            # Simplified sizing for this example
            #qty = risk.desired_size(cfg["symbol"], row["open"])

            #if qty != 0:
            #    broker.queue_order(cfg["symbol"], qty)

            # Use the EVEngine’s Kelly‐fraction to size the trade:
            # Allocate position_size × (current equity) in dollars,
            # then convert to a share quantity at next‐bar open.
            current_cap = risk.account_equity + mtm
            dollar_allocation = ev_res.position_size * current_cap
            qty = int(dollar_allocation // row["open"])

            if qty > 0:
                broker.queue_order(cfg["symbol"], qty)

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
                "residual": ev_res.residual,
                "position_size": ev_res.position_size,
                "open": row["open"],
                "close": row["close"],
                "cluster_id": ev_res.cluster_id,
            })

        log.info("Backtest loop finished.")
        return pd.DataFrame(equity_curve)