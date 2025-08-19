# scripts/diagnostics.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from prediction_engine.market_regime import label_days, RegimeParams
from pathlib import Path
from prediction_engine.calibration import map_mu_to_prob


from prediction_engine.calibration import map_mu_to_prob, load_calibrator

class BacktestDiagnostics:
    def __init__(self, df: pd.DataFrame, raw: pd.DataFrame, cfg: dict):
        self.df  = df.copy()
        self.raw = raw.copy()
        self.cfg = cfg

        self.df["ret1m"]      = self.df["open"].shift(-1) / self.df["open"] - 1
        self.df["cum_return"] = self.df["equity"] / cfg["equity"] - 1
        self.df["is_entry"]   = (self.df.get("qty_next_bar", 0) > 0)

        self.calib_dir = None
        calib_path = (cfg or {}).get("calibration_dir") or (cfg or {}).get("calib_dir")
        if calib_path:
            self.calib_dir = Path(calib_path)

        '''# --- compute p ONLY for entry bars; NaN elsewhere ------------------
        p = np.full(len(self.df), np.nan, dtype=float)
        entry_idx = self.df.index[self.df["is_entry"]]
        if len(entry_idx) > 0:
            mu_vec = self.df.loc[entry_idx, "mu"].to_numpy()
            p_vals = map_mu_to_prob(
                mu_vec,
                calibrator=None,                 # let loader handle it
                artefact_dir=self.calib_dir,     # uses on-disk iso if present
                default_if_missing=None,         # else smooth fallback
            )
            if p_vals.shape[0] != entry_idx.shape[0]:
                raise ValueError(
                    f"[Diagnostics] p length {p_vals.shape[0]} != #entries {entry_idx.shape[0]}"
                )
            print("mu[:5] =", mu_vec[:5])
            print("p[:5]  =", p_vals[:5])

            p[self.df["is_entry"].to_numpy()] = p_vals
        self.df["p"] = p'''

        # --- calibrated probability p: prefer precomputed values from df ---
        # We want p only on ENTRY bars; NaN elsewhere.
        entries_mask = self.df["is_entry"].to_numpy()

        # If caller already provided p (e.g., run_backtest.py), preserve it for entries
        has_p_col = ("p" in self.df.columns) and self.df["p"].notna().any()
        if has_p_col:
            # keep given p on entries; force non-entries to NaN
            self.df.loc[~self.df["is_entry"], "p"] = np.nan
        else:
          # compute p ONLY for entries; leave others NaN
            p = np.full(len(self.df), np.nan, dtype=float)
            entry_idx = self.df.index[self.df["is_entry"]]
            if len(entry_idx) > 0:
                mu_vec = self.df.loc[entry_idx, "mu"].to_numpy(dtype=float)
                # Try to load on-disk calibrator; otherwise map_mu_to_prob falls back smoothly
                try:
                    p_vals = map_mu_to_prob(
                        mu_vec,
                        calibrator = None,  # loader handles on-disk if available
                        artefact_dir = self.calib_dir,  # uses on-disk iso if present
                        default_if_missing = None,  # else monotone fallback
                    )
                except Exception as e:
                    print(f"[Diagnostics] map_mu_to_prob failed ({e}); using neutral mapping.")
                    # Neutral, rank-preserving fallback (same idea as map_mu_to_prob’s tanh)
                    finite = np.isfinite(mu_vec)
                    scale = np.median(np.abs(mu_vec[finite])) * 5.0 if finite.any() else 1.0
                    p_vals = 0.5 * (1.0 + np.tanh(mu_vec / max(scale, 1e-3)))

                # Guardrail: clip p away from 0 and 1 to avoid saturation on tiny samples
                #p_vals = np.clip(p_vals, 0.35, 0.65)


                if p_vals.shape[0] != entry_idx.shape[0]:

                    raise ValueError(
                        f"[Diagnostics] p length {p_vals.shape[0]} != #entries {entry_idx.shape[0]}"
                    )
                print("mu[:5] =", mu_vec[:5])
                print("p[:5]  =", p_vals[:5])
                p[self.df["is_entry"].to_numpy()] = p_vals
                self.df["p"] = p

        '''# after self.df["p"] is computed
        entries = (self.df["qty_next_bar"] > 0)
        p_non_null = self.df["p"].notna().sum()
        print(f"[Diag] entries={entries.sum()} / rows={len(self.df)}")
        print(
            f"[Diag] p non-null={p_non_null}  min/max(valid)={self.df['p'].dropna().min():.3f}/{self.df['p'].dropna().max():.3f}")

        u = np.unique(self.df.loc[entries, "p"].dropna().values)
        print(f"[Diag] unique p on entries: count={len(u)}  sample={u[:10]}")

        # Optional: quick sanity prints
        print(f"[Diag] entries={self.df['is_entry'].sum()} / rows={len(self.df)}")
        print(f"[Diag] p non-null={self.df['p'].notna().sum()}  "
              f"min/max(valid)={np.nanmin(self.df['p']):.3f}/{np.nanmax(self.df['p']):.3f}")'''

        # Prints that don’t crash if there are zero valid p’s
        entries = (self.df["qty_next_bar"] > 0)
        p_non_null = int(self.df["p"].notna().sum())
        print(f"[Diag] entries={entries.sum()} / rows={len(self.df)}")
        if p_non_null > 0:
            pmin = float(self.df["p"].dropna().min())
            pmax = float(self.df["p"].dropna().max())
            print(f"[Diag] p non-null={p_non_null}  min/max(valid)={pmin:.3f}/{pmax:.3f}")
            u = np.unique(self.df.loc[entries, "p"].dropna().values)
            print(f"[Diag] unique p on entries: count={len(u)}  sample={u[:10]}")
        else:
            print(f"[Diag] p non-null=0")

    def print_trade_stats(self):
        trades = self.df["qty_next_bar"] > 0
        wins   = (self.df["ret1m"] > 0) & trades
        avg_ret = self.df.loc[trades, "ret1m"].mean()
        print(f"\nPositive‑µ trade stats:\n"
              f"  Total trades: {trades.sum()}\n"
              f"  Win rate:    {wins.sum()/trades.sum():.3f}\n"
              f"  Avg ret:     {avg_ret:.4f}\n")

    def calibration(self):
        if self.df.empty:
            print("[CAL] input df empty – skipping calibration");
            return

        # µ deciles (all rows)
        self.df["decile"] = pd.qcut(self.df["mu"], 10, labels=False, duplicates="drop")
        bucket = self.df.groupby("decile")["ret1m"].mean()
        win_by_decile = self.df.groupby("decile")["ret1m"].apply(lambda s: (s > 0).mean())

        print("Win-rate by µ decile:\n", win_by_decile.to_string())
        print("[CAL] bin counts:", self.df.groupby("decile")["mu"].count().to_dict())
        print("=== Calibration (µ-decile → avg 1-bar return) ===")
        print(bucket.to_string(), "\n")

        ## p-deciles (entries only → non-NaN p)
        dfp = self.df[self.df["p"].notna()].copy()
        if len(dfp) == 0:
            print("=== Calibration (p-decile) ===\n(no entry bars with valid p)\n")
            return

        dfp["p_decile"] = pd.qcut(dfp["p"], 10, labels=False, duplicates="drop")
        bucket_p = dfp.groupby("p_decile")["ret1m"].mean()
        win_by_p = dfp.groupby("p_decile")["ret1m"].apply(lambda s: (s > 0).mean())

        print("Win-rate by calibrated-p decile:\n", win_by_p.to_string())
        print("[CAL] bin counts (p):", dfp.groupby("p_decile")["p"].count().to_dict())
        print("=== Calibration (p-decile → avg 1-bar return) ===")
        print(bucket_p.to_string(), "\n")

        # keep legacy µ-deciles print too if you like

    def roc_auc(self):
        if self.df.empty:
            print("[AUC] input df empty – skipping ROC");
            return
        dfp = self.df[self.df["p"].notna()].copy()
        if dfp.empty:
            print("[AUC] no valid p on entry bars – skipping ROC");
            return
        y_true = (dfp["ret1m"] > 0).astype(int).to_numpy()
        y_score = dfp["p"].to_numpy()
        auc = roc_auc_score(y_true, y_score)
        # simple µ>0 classifier just for reference (use p if you prefer):
        y_pred = (y_score > 0.5).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        print(f"=== ROC AUC (p vs actual sign on entries) ===\nAUC = {auc:.3f}")
        print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}\n")

    def drawdown(self):
        if self.df.empty:
            print("[DD] input df empty – skipping drawdown");
            return

        eq = self.df["equity"]
        peak = eq.cummax()
        dd   = (eq / peak - 1) * 100
        max_dd = dd.min()
        print(f"=== Drawdown ===\nMax drawdown = {max_dd:.2f}%\n")
        plt.figure(); dd.plot(title="Drawdown (%)"); plt.show()

    def rolling_sharpe(self, window=20):
        rets = self.df["equity"].pct_change().fillna(0)
        rs = (rets.rolling(window).mean() / rets.rolling(window).std()) * np.sqrt(252)
        print("=== Rolling 20‑bar Sharpe ===")
        print(rs.describe().to_string(), "\n")
        plt.figure(); rs.plot(title="Rolling Sharpe"); plt.show()

    def residuals_summary(self):
        print("=== Residuals Summary ===")
        print(self.df["residual"].describe().to_string(), "\n")
        plt.figure(); self.df["residual"].hist(bins=50); plt.title("Residuals"); plt.show()

    def trade_pnl_hist(self):
        col = "realized_pnl"
        if col not in self.df and "realised_pnl" in self.df:
            self.df[col] = self.df["realised_pnl"]
        if col not in self.df:
            print("No 'realized_pnl' column; skipping trade-level P&L histogram.\n")
            return

        if "realized_pnl" not in self.df:
            print("No 'realized_pnl' column; skipping trade‑level P&L histogram.\n")
            return
        pnl = self.df["realized_pnl"]
        print("=== Trade‑level P&L Summary ===")
        print(pnl.describe().to_string(), f"\nWin‑rate = {(pnl>0).mean():.3f}\n")
        plt.figure(); pnl.hist(bins=50); plt.title("Trade‑level P&L"); plt.show()

    def regime_performance(self):
        # recompute daily regimes from raw
        r = self.raw.copy()
        r["timestamp"] = pd.to_datetime(r["Date"] + " " + r["Time"])
        r.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}, inplace=True)
        daily = r.set_index("timestamp").resample("D").agg({
            "open":"first","high":"max","low":"min","close":"last"
        }).dropna()
        regimes = label_days(daily, RegimeParams())
        self.df["regime"] = pd.to_datetime(self.df["timestamp"]).dt.normalize().map(regimes)
        cum_by_regime = self.df.groupby("regime")["equity"] \
            .apply(lambda x: x.iloc[-1]/x.iloc[0] - 1)
        print("=== Cumulative Return by Regime ===")
        print(cum_by_regime.to_string(), "\n")
        perf = self.df.groupby("regime")["cum_return"].agg(["min","max","mean"])
        print("=== Regime performance summary ===")
        print(perf.to_string(), "\n")
        plt.figure(); cum_by_regime.plot(kind="bar", title="Cum return by regime"); plt.show()

    def cluster_exposure(self):
        counts = self.df["cluster_id"].value_counts().sort_index()
        pnl    = self.df.groupby("cluster_id")["realized_pnl"].sum() if "realized_pnl" in self.df else None
        #table  = pd.DataFrame({"count": counts, "total_pnl": pnl}).fillna(0)
        table = pd.DataFrame({"count": counts, "total_pnl": pnl})
        # forward-compatible with pandas deprecations
        table = table.infer_objects(copy=False).fillna(0)
        top10  = table.sort_values("count", ascending=False).head(10)
        print("=== Top 10 Clusters by Trade Count ===")
        print(top10.to_string(), "\n")
        plt.figure(); top10["count"].plot(kind="bar", title="Trades per cluster"); plt.show()

    def run_all(self):
        self.print_trade_stats()
        self.calibration()
        self.roc_auc()
        self.drawdown()
        self.rolling_sharpe()
        self.residuals_summary()
        self.trade_pnl_hist()
        self.regime_performance()
        self.cluster_exposure()
