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
        for i, (df_index, row) in enumerate(self.feats_df.iterrows(), 1):
            '''bar = {
                "ts": row["timestamp"],
                "symbol": row["symbol"],
                "price": row["open"],  # Fills at NEXT bar's open
            }'''

            # --- derive live half-spread -------------------------------------------------
            # NB: raw bars have only O/H/L/C; use intrabar range as a crude proxy.
            # Typical minute-bar low/high spread on small-caps ≈ 0.05–0.15 %.
            half_spread = max(1e-6, (row["high"] - row["low"]) * 0.5)

            bar = {
                "ts": row["timestamp"],
                "symbol": row["symbol"],
                "price": row["open"],  # next-bar open fill
                "half_spread": half_spread,  # ←  NEW – pass into EVEngine → cost model
                }

            # --- proactive stop‑loss / take‑profit -----------------
            # --- proactive stop‑loss / take‑profit (flag only!) -------------
            sl_hit = False
            if risk.position_size > 0:
                entry_px = risk.avg_entry_price
                if (row["low"] <= entry_px * (1 - self.STOP_LOSS_PCT) or
                        row["high"] >= entry_px * (1 + self.TAKE_PROFIT_PCT)):
                    sl_hit = True  # we’ll flatten later

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

                #risk.position_size = tr["qty"] + risk.position_size



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

            # -------- EV evaluation ---------------------------------
            adv_pct = row.get("adv_pct", 10.0)

            #half_spread = row.get("half_spread", 0.01)

            raw_spread = (row["high"] - row["low"]) * 0.5
            # minute-bar range is much wider than bid-ask; assume only 10 % of it
            half_spread = min(0.002 * row["close"], max(0.0002, 0.02 * raw_spread)) # 2 % of bar‐range<br>
            #  ─ determine today's regime (may return NaN / None) ─
            reg = self.regimes.get(row["timestamp"].normalize(), None)
            if pd.isna(reg):
                reg = None  # keep evaluate() happy

            ev_raw = ev.evaluate(
                x_vec,
                half_spread = half_spread,
                adv_percentile = adv_pct,
                regime = reg,
            )

            print("[DBG] residual=", ev_raw.residual)

            # skip permanently bad clusters
            #if ev_raw.cluster_id not in self.good_clusters:
            #    continue

            if ev_raw.cluster_id not in self.good_clusters:
                continue

            #if ev_raw.residual > self.ev.residual_threshold:
            #    continue

            # ░░░░░░░░░░  PATCH 1  ░░ Residual gate + size ∝ μ ░░░░░░░░░░
            # === paste AFTER the line that builds `ev_raw` (look for:  ev_raw = ev.evaluate(...)) ===

            # ── 1. Skip high‑residual analogues ────────────────────────
            if ev_raw.residual > self.ev.residual_threshold:
                # “too far” from any centroid; ignore this bar
                continue

            # ── 2. Continuous μ → expected return (already calibrated) ─
            exp_ret = float(self.cal.predict([[ev_raw.mu]])[0]) if self.cal else ev_raw.mu

            '''# ── 3. Position sizing proportional to expected return ─────
            #     min_cap_kelly ≈ 0.30 prevents tiny noisy trades
            min_cap_kelly = 0.30
            pos_fraction = max(0.0, exp_ret / 0.01)  # 1 % exp ret ⇒ full Kelly
            pos_fraction = np.clip(pos_fraction, 0.0, 1.0)  # cap at 100 %
            pos_fraction = max(pos_fraction, min_cap_kelly) if pos_fraction > 0 else 0.0

            qty = risk.kelly_position(
                mu=exp_ret,
                variance_down=ev_raw.variance_down,
                price=row["open"],
                adv=row.get("adv", None),
                override_frac=pos_fraction,  # <── NEW ARG your RiskManager already supports
            )'''

            # ── 3.  Probability‑weighted Kelly fraction ──────────────────────
            #     Edge = P(up)‑0.55 ; full size once edge ≥ 20 %
            #p_up = float(self.cal.predict([[ev_raw.mu]])[0]) if self.cal else 0.5
            # crude mapping: assume N(0,σ)   P(up)≈Φ(μ/σ)
            p_up = 0.5 * (1.0 + math.erf(ev_raw.mu / (2.0 * (ev_raw.sigma ** 0.5 + 1e-9))))

            edge = max(0.0, p_up - 0.55)  # need > 55 % to trade
            pos_fraction = np.clip(edge / 0.20, 0.0, 1.0)  # full Kelly at 75 %

            qty = risk.kelly_position(
                mu=exp_ret,
                variance_down=ev_raw.variance_down,
                price=row["open"],
                adv=row.get("adv", None),
                override_frac=pos_fraction,
            )

            # =====================================================================

            mu_cal = (
                float(self.cal.predict([[ev_raw.mu]])[0])
                if self.cal is not None else ev_raw.mu
                )




            ev_res = ev_raw.__class__(
                mu = ev_raw.mu,
                sigma = ev_raw.sigma,
                variance_down = ev_raw.variance_down,
                beta = ev_raw.beta,
                residual = ev_raw.residual,
                cluster_id = ev_raw.cluster_id,
                outcome_probs = ev_raw.outcome_probs,
                position_size = ev_raw.position_size,
                drift_ticket = ev_raw.drift_ticket,
                        )

            #ev_res = ev_raw



            # -------- Kelly sizing ----------------------------------
            adv_today = row.get("adv", None)

            #if _ := ev_res.mu > 0:
            #    print(f"[DBG] bar {bar['ts']} µ={ev_res.mu:.5f}  cost_ps="
             #         f"{risk.cost_model.estimate():.4f}  cluster={ev_res.cluster_id}")



            '''if ev_res.mu > 0:  # only long on positive µ
                qty = risk.kelly_position(
                mu = ev_res.mu,
                variance_down = ev_res.variance_down,
                price = row["open"],
                adv = adv_today,
                )
            else:
                qty = 0'''

            '''# ── C: Gate on calibrated probability, not just sign ─────────────
            if self.cal is not None:
                p_win = float(self.cal.predict([[ev_res.mu]])[0])
            else:
                p_win = 0.0
            # only take a position when confidence > 55%
            if p_win > 0.55:
                qty = risk.kelly_position(
                    mu = ev_res.mu,
                    variance_down = ev_res.variance_down,
                    price = row["open"],
                    adv = adv_today,
                )
            else:
                qty = 0'''

            # ── SIMPLIFIED: Gate purely on raw µ>0 ──────────────────────────
            '''if ev_res.mu > 0:
                qty = risk.kelly_position(
                    mu = ev_res.mu,
                    variance_down = ev_res.variance_down,
                    price = row["open"],
                    adv = adv_today,
                )
            else:
                qty = 0'''
            # -- gate out bad analogues EARLY -----------------------
            #if ev_raw.residual > self.ev.residual_threshold:
            #    continue

            # ------------------------------------------------------------------
            # Use regression calibrator:  μ → expected 1‑bar return (E[r₁])
            # ------------------------------------------------------------------
            #exp_ret = (float(self.cal.predict([[ev_raw.mu]])[0])
            #           if self.cal is not None else ev_raw.mu)
            exp_ret = ev_raw.mu


            '''if exp_ret <= 0:          # no positive edge
                qty = 0
            else:
                qty = risk.kelly_position(
                          mu             = exp_ret,
                          variance_down  = ev_raw.variance_down,
                          price          = row["open"],
                          adv            = adv_today,
                          #override_frac  = pos_fraction
                       )'''

            # --- gate out weak edge OR bad analogue ----------------------------
            _min_edge = 0.0004  # 4 bp ~ half today’s spread; tune if needed
            if (exp_ret < _min_edge) or (ev_raw.residual > 0.75):
                qty = 0
            else:
                qty = risk.kelly_position(
                    mu=exp_ret,
                    variance_down=ev_raw.variance_down,
                    price=row["open"],
                    adv=adv_today,
                )

            '''#ev_res = ev.evaluate(x_vec, adv_percentile=10.0, regime=None, half_spread=0)

            adv_pct = row.get("adv_pct", 10.0)
            half_spread = row.get("half_spread", 0.01)  # $0.01 fallback

            ev_raw = ev.evaluate(
                x_vec,
                half_spread = half_spread,
                adv_percentile = adv_pct,
                regime = None,
            )

              # skip clusters that systematically lose money

            if ev_raw.cluster_id not in self.good_clusters:
                continue
            mu_cal = self.cal.predict([[ev_raw.mu]])[0] if self.cal else ev_raw.mu

            ev_res = ev_raw.__class__(
                mu = mu_cal,
                sigma = ev_raw.sigma,
                variance_down = ev_raw.variance_down,
                beta = ev_raw.beta,
                residual = ev_raw.residual,
                cluster_id = ev_raw.cluster_id,
                outcome_probs = ev_raw.outcome_probs,
                position_size = ev_raw.position_size,
                drift_ticket = ev_raw.drift_ticket,
                )


            qty = risk.kelly_position(
            qty = risk.kelly_position(
                mu=ev_res.mu,

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
                qty = 0'''


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
            # ─── exit sizing (1‑bar hold *or* stop‑loss) ───────────────────
            if sl_hit and risk.position_size > 0:
                # stop‑loss or take‑profit just triggered
                qty_exit = -risk.position_size
            elif risk.position_size > 0:
                # 1‑bar hold rule (optional – comment out to let winners run)
                qty_exit = -risk.position_size
            else:
                qty_exit = 0

            # ░░░░░░░░░░  PATCH 2  ░░ 1‑bar stop‑loss / take‑profit ░░░░░░░░░░
            # === paste RIGHT BEFORE the section that queues `qty_exit` / `qty` orders ===
            #     (search for the comment "# queue orders for next bar --------------------------------------------")

            # --- intrabar risk control (executed on NEXT bar open) -----------------
            '''if risk.position_size > 0:
                #entry_px = broker.average_entry_price(cfg["symbol"])
                entry_px = risk.avg_entry_price

                stop_px = entry_px * (1 - 0.003)  # 30 bp stop‑loss
                tp_px = entry_px * (1 + 0.006)  # 60 bp take‑profit

                if row["open"] <= stop_px or row["open"] >= tp_px:
                    qty_exit = -risk.position_size
            else:
                qty_exit = 0'''
            # ──────────────────────────────────────────────────────────────────────────

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

            # ─── DIAG: every 50 bars ───────────────────────────────────────────
            if i % 50 == 0:
                cur_eq = risk.account_equity + mtm
                max_dd = 1.0 - cur_eq / max(x["equity"] for x in equity_curve) \
                    if equity_curve else 0.0
                print(f"[Snap] bar={i:4d}  eq={cur_eq:,.0f}  "
                      f"open_pos={risk.position_size}  maxDD={max_dd:.4f}")
            # ───────────────────────────────────────────────────────────────────

            equity_curve.append({
                "timestamp": bar["ts"],
                "equity": current_equity,
                "realised_pnl": realised_pnl,
                "unrealised_pnl": mtm,
                "qty_next_bar": qty,
                "mu_cal": ev_res.mu,
                "sigma_sq": ev_res.sigma,
                "var_down": ev_res.variance_down,  # NEW
                "residual": ev_res.residual,
                "position_size": ev_res.position_size,
                "open": row["open"],
                "close": row["close"],
                "cluster_id": ev_res.cluster_id,
                "notional_abs": abs(risk.position_size * row["open"]),  #New
                "adv_used": adv_today,

            })




        log.info("Backtest loop finished.")

        equity_df = pd.DataFrame(equity_curve)  # ← you create this just before return

        # ─── DIAG: end-of-run summary ──────────────────────────────────────
        print("\n[Equity.describe()]")
        print(equity_df["equity"].describe())

        # realised trade-level PnL array
        trade_pnls = equity_df["realised_pnl"].values
        print("\n[Trade-level PnL] n=", len(trade_pnls),
              "  mean=", trade_pnls.mean().round(2),
              "  std=", trade_pnls.std().round(2))

        # quick histogram to CSV for later plotting
        hist_path = Path("backtests") / "trade_pnl_hist.csv"
        hist_path.parent.mkdir(exist_ok=True)
        pd.Series(trade_pnls).to_csv(hist_path, index=False)
        print("[Saved] trade-pnl histogram data →", hist_path)
        # ───────────────────────────────────────────────────────────────────

        return pd.DataFrame(equity_curve)