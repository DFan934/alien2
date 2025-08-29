import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from prediction_engine.calibration import calibrate_isotonic, load_calibrator

def test_isotonic_is_monotone_and_nondegenerate():
    rng = np.random.default_rng(42)
    x = np.linspace(-2, 2, 200)
    y = (x + rng.normal(0, 0.3, size=x.size) > 0).astype(float)

    with TemporaryDirectory() as d:
        iso, path = calibrate_isotonic(x, y, out_dir=Path(d))
        iso2 = load_calibrator(Path(d))
        pred = iso2.predict(x)
        # monotone non-decreasing
        assert (np.diff(pred) >= -1e-12).all()
        # not collapsed to a constant
        assert pred.max() - pred.min() > 1e-3
