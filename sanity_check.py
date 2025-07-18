from pathlib import Path
import numpy as np
from prediction_engine.ev_engine import EVEngine
from prediction_engine.tx_cost import BasicCostModel

art   = Path("weights")
cost  = BasicCostModel()
ev    = EVEngine.from_artifacts(art, cost_model=cost)

# load the “ground‑truth” cluster means
stats = np.load(art/"cluster_stats.npz", allow_pickle=True)
centers = np.load(art/"centers.npy")

print("cluster_id  µ_cluster   µ_kernel   µ_synth   residual")
for c in centers[:20]:
    r = ev.evaluate(c, half_spread=0)
    true_mu   = stats["mu"][r.cluster_id]
    # if your EVResult has attributes named mu_kernel and mu_synth:
    kern_mu   = getattr(r, "mu_kernel", float("nan"))
    synth_mu  = getattr(r, "mu_synth",  float("nan"))
    print(f"{r.cluster_id:>10d}   "
          f"{true_mu:>8.5f}   "
          f"{kern_mu:>8.5f}   "
          f"{synth_mu:>8.5f}   "
          f"{r.residual:>7.3f}")
