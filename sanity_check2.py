from pathlib import Path
import numpy as np, pandas as pd
from prediction_engine.ev_engine import EVEngine
from prediction_engine.tx_cost import BasicCostModel

art = Path("weights")
ev  = EVEngine.from_artifacts(art, cost_model=BasicCostModel())   # cost‑aware

stats = np.load(art/"cluster_stats.npz", allow_pickle=True)
mu_c = stats["mu"]
print("μ centroid summary:\n", pd.Series(mu_c).describe())
print("μ > 0 fraction:", (mu_c > 0).mean())
