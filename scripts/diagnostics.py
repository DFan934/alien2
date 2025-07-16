# scripts/diagnostics.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from prediction_engine.market_regime import label_days, RegimeParams

class BacktestDiagnostics:
    def __init__(self, df: pd.DataFrame, raw: pd.DataFrame, cfg: dict):
        """
        df: the equity‐curve DataFrame returned by Backtester.run()
        raw: the original raw bar DataFrame (with open, high, low, close, volume, timestamp)
        cfg: the backtest CONFIG dict
        """
        self.df = df.copy()
        self.raw = raw.copy()
        self.cfg = cfg

        # core series
        self.df["ret1m"] = self.df["open"].shift(-1) / self.df["open"] - 1
        self.df["cum_return"] = self.df["equity"] / cfg["equity"] - 1

    def print_trade_stats(self):
        trades = self.df["qty_next_bar"] > 0
        wins   = (self.df["ret1m"] > 0) & trades
        avg_ret = self.df.loc[trades, "ret1m"].mean()
        print(f"\nPositive‑µ trade stats:\n"
              f"  Total trades: {trades.sum()}\n"
              f"  Win rate:    {wins.sum()/trades.sum():.3f}\n"
              f"  Avg ret:     {avg_ret:.4f}\n")

    def calibration(self):
        self.df["decile"] = pd.qcut(self.df["mu"], 10, labels=False, duplicates="drop")
        bucket = self.df.groupby("decile")["ret1m"].mean()
        print("=== Calibration (µ‑decile → avg 1‑bar return) ===")
        print(bucket.to_string(), "\n")

    def roc_auc(self):
        # μ > 0 as predictor of ret1m>0
        y_true  = (self.df["ret1m"] > 0).astype(int)
        y_score = self.df["mu"].values
        auc = roc_auc_score(y_true, y_score)
        print(f"=== ROC AUC (µ>0 vs actual sign) ===\nAUC = {auc:.3f}\n")

    def drawdown(self):
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
        table  = pd.DataFrame({"count": counts, "total_pnl": pnl}).fillna(0)
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
