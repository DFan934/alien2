#/prediction_engine/calibration.py

"""
Isotonic‑regression calibration for µ‑scores.
Fits on a label vector (1 = positive return, 0 = negative/zero) and
saves a pickled `IsotonicRegression` object.
"""

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

__all__ = ("calibrate_isotonic", "load_calibrator")


# --------------------------------------------------------------------- #
# Fit & persist                                                         #
# --------------------------------------------------------------------- #
def calibrate_isotonic(
    mu_vals: np.ndarray,
    labels: np.ndarray,
    out_dir: Path,
    *,
    y_min: float = 0.0,
    y_max: float = 1.0,
) -> Tuple[IsotonicRegression, Path]:
    """
    Parameters
    ----------
    mu_vals : np.ndarray
        Raw µ‑scores from EVEngine (shape = [n_samples]).
    labels  : np.ndarray
        Binary outcomes (1 = win, 0 = loss) aligned with *mu_vals*.
    out_dir : Path
        Directory where ``iso_calibrator.pkl`` will be written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=y_min, y_max=y_max)
    iso.fit(mu_vals.astype(float), labels.astype(float))

    # --- sanity: mapping must not collapse to a constant -------------------
    if np.ptp(iso.predict(mu_vals)) < 1e-4:
        raise RuntimeError(
                        "Isotonic calibration collapsed to a flat line – check label mix!"
            )

    print(out_dir)
    f = out_dir / "iso_calibrator.pkl"
    print(f)
    joblib.dump(iso, f)
    return iso, f


# --------------------------------------------------------------------- #
# Inference helper                                                      #
# --------------------------------------------------------------------- #
def load_calibrator(artefact_dir: Path) -> IsotonicRegression:
    """Return the previously‑saved isotonic regressor."""
    return joblib.load(artefact_dir / "iso_calibrator.pkl")