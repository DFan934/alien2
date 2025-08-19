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

from prediction_engine.calibration import map_mu_to_prob
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
    # ─── strategy knobs (tune here) ────────────────────────────────
    STOP_LOSS_PCT   = 0.01   # 1 % below entry
    TAKE_PROFIT_PCT = 0.015  # 1.5 % above entry
    MAX_KELLY_PROB  = 0.60   # minimum calibrated edge to size up


    def __init__(self,
                 features_df: pd.DataFrame,
                 cfg: Dict[str, Any],
                 *,
                 ev: EVEngine,
                 risk: RiskManager,
                 broker: Any,
                 cost_model: BasicCostModel,
                 calibrator,
                 good_clusters: set[int],
                 regimes: pd.Series,
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
        self.cal = calibrator
        self.regimes = regimes

    def run(self) -> pd.DataFrame:
        """
        Executes the main backtesting event loop.
        """



        cfg = self.cfg
        ev = self.ev
        risk = self.risk
        broker = self.broker
        log = self.log

        reasons = dict(
            not_good_cluster=0,
            residual_gate=0,
            p_gate=0,
            qty_zero=0,
            took_entry=0,
        )

        equity_curve: List[Dict[str, Any]] = []
        log.info("Starting backtest with %d bars...", len(self.feats_df))

        # --- knobs (lightly tuneable) ---
        P_GATE = float(cfg.get("p_gate", 0.51))
        FULL_P = float(cfg.get("full_p", 0.61))
        p_span = max(1e-6, FULL_P - P_GATE)

        log.info("Gates: P_GATE=%.3f  FULL_P=%.3f", P_GATE, FULL_P)

        for i, (_, row) in enumerate(self.feats_df.iterrows(), 1):
            # --- BAR meta (we fill at next bar's open via BrokerStub) ---
            bar = {
                "ts": row["timestamp"],
                "symbol": row["symbol"],
                "price": row["open"],  # next open
                "half_spread": 0.0,  # free-mode costs → keep 0
            }

            # --- execute any pending orders from prior bar ---
            fills = broker.execute_pending(bar)
            realised_pnl = 0.0
            for tr in fills:
                fill_dict = {
                    "price": tr["price"],
                    "size": abs(tr["qty"]),
                    "side": "BUY" if tr["qty"] > 0 else "SELL",
                    "trade_id": tr.get("trade_id", ""),
                }
                is_closed, pnl, _ = risk.process_fill(fill_dict)
                if not math.isfinite(pnl):
                    raise RuntimeError(f"Bad realized_pnl: {pnl!r} for fill={fill_dict!r} ts={bar['ts']}")
                if not math.isfinite(risk.account_equity):
                    raise RuntimeError(f"Non-finite equity after fill on {bar['ts']}")
                realised_pnl += pnl

            # mark-to-market on current bar’s close
            mtm = broker.mark_to_market(bar)

            # --- build EV input vec & regime for this timestamp ---
            x_vec = row[self.pc_cols].to_numpy(dtype=np.float32)
            reg = self.regimes.get(row["timestamp"].normalize(), None)
            if pd.isna(reg):
                reg = None

            # --- evaluate EVEngine (no costs; ADV percentile only for sizing stats) ---
            ev_raw = ev.evaluate(
                x_vec,
                half_spread=0.0,
                adv_percentile=10.0,
                regime=reg,
            )

            # after ev_raw = ev.evaluate(...)
            if ev_raw.cluster_id not in self.good_clusters:
                reasons["not_good_cluster"] += 1
                continue

            if ev_raw.residual > self.ev.residual_threshold:
                reasons["residual_gate"] += 1
                continue

            # ── hard gates: cluster + residual ----------------------------------
            #if ev_raw.cluster_id not in self.good_clusters:
                # skip systematically weak clusters
            #    continue
            #if ev_raw.residual > self.ev.residual_threshold:
                # analogue too far from any centroid
            #    continue


            print("[Skip reasons]", reasons)

            # ── probability gate (calibrated in run_backtest via ev._calibrator) --
            # ev_raw.p_up is provided by EVEngine if ._calibrator is set
            '''p_up = float(getattr(ev_raw, "p_up", 0.5))
            if p_up < P_GATE:
                qty = 0
            else:
                # map probability edge → fraction of equity (capped by max_kelly)
                edge = (p_up - P_GATE) / p_span  # 0 .. 1 when p ∈ [P_GATE, FULL_P]
                frac = float(np.clip(edge, 0.0, 1.0)) * risk.max_kelly

                # expected return proxy (you can swap to iso_ret if you pass it in)
                exp_ret = ev_raw.mu

                # use ADV (shares) for the liquidity brake inside RiskManager
                adv_today = row.get("adv_shares", None)

                # Kelly with probability-scaled override fraction
                qty = risk.kelly_position(
                    mu=exp_ret,
                    variance_down=ev_raw.variance_down,
                    price=row["open"],
                    adv=adv_today,
                    override_frac=frac,
                )'''

            ## robust p_up: use in-memory calibrator; fall back to a smooth CDF if missing
            #if self.cal is not None:
            #    p_up = float(np.clip(self.cal.predict([[ev_raw.mu]])[0], 0.35, 0.65))
            #else:
            #    # safe fallback so we still get variation if calibrator isn't present
            #    denom = max(1e-6, float(ev_raw.sigma)) ** 0.5
            #    p_up = 0.5 * (1.0 + math.erf(ev_raw.mu / (2.0 * denom)))

            # --- probability & sizing (ordered!) ---
            # --- probability & sizing (ordered!) ---
            # Prefer the engine’s calibrated probability (already using iso_prob)
            p_up = float(getattr(ev_raw, "p_up", float("nan")))
            # In Backtester.run() before sizing:
            if self.cfg.get("force_entries_on_mu", False):
                p_up = 0.55 if ev_raw.mu > 0 else 0.45

            if not math.isfinite(p_up):
                if self.cal is not None:
                    p_up = float(self.cal.predict([ev_raw.mu])[0])
                else:
                    denom = max(1e-6, float(ev_raw.sigma)) ** 0.5
                    p_up = 0.5 * (1.0 + math.erf(ev_raw.mu / (2.0 * denom)))

            # CLIP HERE so the gate has a fighting chance
            p_up = float(np.clip(p_up, 0.30, 0.70))

            # DO NOT clamp here; let the gate decide
            if p_up < P_GATE:
                qty = 0
                reasons["p_gate"] += 1
            else:
                edge = (p_up - P_GATE) / p_span
                frac = float(np.clip(edge, 0.0, 1.0)) * risk.max_kelly
                exp_ret = ev_raw.mu  # use raw μ for sizing target (you already subtract cost elsewhere)
                adv_today = row.get("adv_shares", None)
                qty = risk.kelly_position(
                    mu=exp_ret,
                    variance_down=ev_raw.variance_down,
                    price=row["open"],
                    adv=adv_today,
                    override_frac=frac,
                )
                if qty <= 0:
                    reasons["qty_zero"] += 1
                else:
                    reasons["took_entry"] += 1

            # TEMP: use gentler gates until we see trades; we can tighten later
            #P_GATE = 0.51
            #FULL_P = 0.61
            #p_span = max(1e-6, FULL_P - P_GATE)

            '''if p_up < P_GATE:
                qty = 0
            else:
                edge = (p_up - P_GATE) / p_span
                frac = float(np.clip(edge, 0.0, 1.0)) * risk.max_kelly
                exp_ret = ev_raw.mu
                adv_today = row.get("adv_shares", None)  # ← use the real ADV shares column
                qty = risk.kelly_position(
                    mu=exp_ret,
                    variance_down=ev_raw.variance_down,
                    price=row["open"],
                    adv=adv_today,
                    override_frac=frac,
                )'''

            # 1-bar stop / take-profit flag you already had
            sl_hit = False
            if risk.position_size > 0:
                entry_px = risk.avg_entry_price
                if (row["low"] <= entry_px * (1 - self.STOP_LOSS_PCT)) or \
                        (row["high"] >= entry_px * (1 + self.TAKE_PROFIT_PCT)):
                    sl_hit = True

            # --- 1-bar hold exit (or immediate stop/TP) ---
            if sl_hit and risk.position_size > 0:
                qty_exit = -risk.position_size
            elif risk.position_size > 0:
                qty_exit = -risk.position_size
            else:
                qty_exit = 0

            # --- queue orders for next bar ---
            if qty_exit != 0:
                broker.queue_order(cfg["symbol"], qty_exit)  # SELL if negative
            elif qty > 0:
                broker.queue_order(cfg["symbol"], qty)  # BUY

            # --- record equity snapshot & diagnostics ---
            current_equity = risk.account_equity + mtm
            if i % 50 == 0:
                cur_eq = current_equity
                max_dd = 1.0 - cur_eq / max(x["equity"] for x in equity_curve) if equity_curve else 0.0
                print(f"[Snap] bar={i:4d}  eq={cur_eq:,.0f}  open_pos={risk.position_size}  maxDD={max_dd:.4f}")

            equity_curve.append({
                "timestamp": bar["ts"],
                "equity": current_equity,
                "realized_pnl": realised_pnl,  # US spelling; diagnostics handles both
                "unrealised_pnl": mtm,
                "qty_next_bar": qty,
                "mu_raw": getattr(ev_raw, "mu_raw", ev_raw.mu),
                "mu_net": ev_raw.mu,
                #"mu_cal": ev_raw.mu,  # keep same column your scripts expect
                "mu_cal": getattr(ev_raw, "mu_raw", ev_raw.mu),  # or drop the field entirely
                "sigma_sq": ev_raw.sigma,
                "var_down": ev_raw.variance_down,
                "residual": ev_raw.residual,
                "position_size": ev_raw.position_size,
                "open": row["open"],
                "close": row["close"],
                "cluster_id": ev_raw.cluster_id,
                "notional_abs": abs(risk.position_size * row["open"]),
                "adv_used": row.get("adv_shares", None),
                "p_up": p_up,
                "frac": float(np.clip((p_up - P_GATE) / p_span, 0.0, 1.0)) if p_up >= P_GATE else 0.0,
            })

        log.info("Backtest loop finished.")

        equity_df = pd.DataFrame(equity_curve)

        print("\n[Equity.describe()]")
        print(equity_df["equity"].describe())

        trade_pnls = equity_df["realized_pnl"].values
        print("\n[Trade-level PnL] n=", len(trade_pnls),
              "  mean=", trade_pnls.mean().round(2),
              "  std=", trade_pnls.std().round(2))

        hist_path = Path("backtests") / "trade_pnl_hist.csv"
        hist_path.parent.mkdir(exist_ok=True)
        pd.Series(trade_pnls).to_csv(hist_path, index=False)
        print("[Saved] trade-pnl histogram data →", hist_path)

        return equity_df
