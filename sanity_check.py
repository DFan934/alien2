from pathlib import Path
import numpy as np
from prediction_engine.ev_engine import EVEngine
from prediction_engine.tx_cost import BasicCostModel

art = Path("weights")                   # <— your artefact dir
free_cost = BasicCostModel()
ev = EVEngine.from_artifacts(art, cost_model=free_cost)

centers = np.load(art/"centers.npy")
mus_centroid = []
for c in centers[:20]:                    # first 20 clusters is enough
    r = ev.evaluate(c, half_spread=0)     # tell it cost is zero
    mus_centroid.append(r.mu)


print("share of μ>0:", np.mean(np.array(mus_centroid) > 0))
