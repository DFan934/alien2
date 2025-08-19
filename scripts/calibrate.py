#==================================
# FILE: scripts/calibrate.py
#==================================
# scripts/calibrate.py
from pathlib import Path
import pandas as pd
import numpy as np
import glob
from prediction_engine.calibration import calibrate_isotonic

# 1) find latest signals CSV
cands = sorted(glob.glob("backtests/signals_*.csv"))
if not cands:
    raise FileNotFoundError("No backtests/signals_*.csv found")
signals_path = cands[-1]
print("Using:", signals_path)

df = pd.read_csv(signals_path)

# 2) ensure mu and open exist
if "mu" not in df.columns and "mu_raw" in df.columns:
    df["mu"] = df["mu_raw"]  # fallback if you stored raw under mu_raw

# match the backtest horizon: enter at next open, exit next close
if "ret1m" not in df.columns:
    next_open  = df["open"].shift(-1)
    next_close = df["close"].shift(-1)
    df["ret1m"] = (next_close / next_open) - 1.0

mu_vals = df["mu"].to_numpy()
labels  = (df["ret1m"] > 0).astype(int).to_numpy()

# Optional denoise: quantile-binning before fitting
q = pd.qcut(pd.Series(mu_vals), 12, duplicates="drop")  # 12 bins works better on tiny sets
mu_bin = pd.Series(mu_vals).groupby(q).mean().to_numpy()
y_bin  = pd.Series(labels).groupby(q).mean().to_numpy()

# If bins are nearly flat, mix with a smooth logistic so iso has slope
if np.ptp(y_bin) < 0.02:
    med   = np.median(mu_bin)
    scale = np.median(np.abs(mu_bin - med)) * 3.0 or 1e-3
    platt = 1.0 / (1.0 + np.exp(-(mu_bin - med) / scale))
    y_bin = 0.5 * y_bin + 0.5 * platt


# 3) fit & persist to weights/calibration
out_dir = Path("../weights") / "calibration"
iso, pkl_path = calibrate_isotonic(mu_bin, y_bin, out_dir)

print("Written calibrator to:", pkl_path)
# quick smoke-test:
qs = np.quantile(mu_vals, [0, 0.25, 0.5, 0.75, 1.0])
import joblib
print("p(mu quantiles):", joblib.load(pkl_path).predict(qs))
