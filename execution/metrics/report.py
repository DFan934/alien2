# ===========================
# metrics/report.py
# ===========================


"""
Quick CLI to produce PnL curve, drawdown stats, and latency summary
from the blotter CSV written by ExecutionManager.
"""
from pathlib import Path
import pandas as pd
import numpy as np

def load_blotter(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=[
        "raw_json", "latency_ms"])
    df["ts"] = pd.to_datetime(df["raw_json"]
                                .str.extract(r'"ts":\s*([\d.]+)')[0]
                                .astype(float), unit="s")
    df["pnl"] = df["raw_json"].str.extract(r'"pnl":\s*([-.\d]+)')[0].astype(float).fillna(0.0)
    return df.sort_values("ts")

def pnl_curve(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cum_pnl"] = out["pnl"].cumsum()
    out["peak"] = out["cum_pnl"].cummax()
    out["drawdown"] = out["cum_pnl"] - out["peak"]
    return out[["ts", "cum_pnl", "drawdown"]]

def latency_summary(df: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "mean_ms": df["latency_ms"].mean(),
        "p95_ms":  np.percentile(df["latency_ms"], 95),
        "max_ms":  df["latency_ms"].max(),
    })

if __name__ == "__main__":
    import argparse, matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("blotter", help="Path to blotter CSV")
    args = ap.parse_args()

    blot = load_blotter(args.blotter)
    print(latency_summary(blot).round(3))

    pnl = pnl_curve(blot)
    plt.plot(pnl["ts"], pnl["cum_pnl"])
    plt.title("Cumulative PnL")
    plt.show()
