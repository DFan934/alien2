# ---------------------------------------------------------------------------
# FILE: tests/test_nightly_calibrate.py
# ---------------------------------------------------------------------------
"""CI smoke test: ensure nightly_calibrate writes four JSON files."""
import sys
from pathlib import Path
import shutil, tempfile, importlib

def test_nightly_calibrate_smoke(monkeypatch):
    tmp = Path(tempfile.mkdtemp())
    (tmp / "data/pnl_by_regime").mkdir(parents=True)

    # --- INSERT HERE, at the start of the function ---
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    # write toy PnL per regime
    import numpy as np, pandas as pd
    for reg in ("trend", "range", "volatile", "global"):
        pnl = pd.Series(np.random.normal(0.001, 0.01, 256))
        pnl.to_csv(tmp / "data/pnl_by_regime" / f"{reg}.csv")

    monkeypatch.chdir(tmp)
    mod = importlib.import_module("scripts2.nightly_calibrate")
    mod.main()

    for reg in ("trend", "range", "volatile", "global"):
        fp = tmp / f"artifacts/weights/regime={reg}/curve_params.json"
        assert fp.exists(), f"missing {fp}"

    shutil.rmtree(tmp, ignore_errors=True)
