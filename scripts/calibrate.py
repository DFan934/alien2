#==================================
# FILE: scripts/calibrate.py
#==================================
from pathlib import Path
import pandas as pd
import numpy as np

from prediction_engine.calibration import calibrate_isotonic

# 1) load your backtest results (signals & realized 1‑bar returns):
df = pd.read_csv("backtest_signals.csv")
mu_vals = df["mu"].to_numpy()
# if ret1m wasn’t in the CSV, compute it from the open prices
if "ret1m" not in df.columns:
    df["ret1m"] = df["open"].shift(-1) / df["open"] - 1.0
labels  = (df["ret1m"] > 0).astype(int).to_numpy()
#labels = (df["ret1m"] > 0).astype(int).to_numpy()

# 2) call calibrate:
out_dir = Path("../weights") / "calibration"
iso, pkl_path = calibrate_isotonic(mu_vals, labels, out_dir)
print("Written calibrator to:", pkl_path)