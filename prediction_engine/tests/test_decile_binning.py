import numpy as np
import pandas as pd
from scripts.diagnostics import BacktestDiagnostics as sd

def test_safe_deciles_near_uniform_with_ties():
    # 5 unique values repeated heavily -> many ties
    vals = np.repeat(np.arange(5, dtype=float), 200)  # length=1000
    s = pd.Series(vals)
    dec = sd.safe_deciles(s, q=10)
    counts = dec.value_counts().sort_index().to_numpy()
    assert counts.min() > 0
    # keep bins within ±25% of each other on this adversarial set
    assert counts.max() / counts.min() <= 1.25
