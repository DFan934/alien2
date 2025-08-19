#/prediction_engine/calibration.py

"""
Isotonic‑regression calibration for µ‑scores.
Fits on a label vector (1 = positive return, 0 = negative/zero) and
saves a pickled `IsotonicRegression` object.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

__all__ = ("calibrate_isotonic", "load_calibrator", "map_mu_to_prob")


# --------------------------------------------------------------------- #
# Fit & persist                                                         #
# --------------------------------------------------------------------- #
'''def calibrate_isotonic(
    mu_vals: np.ndarray,
    labels: np.ndarray,
    out_dir: Path | None = None,
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
    if out_dir is None:
        out_dir = Path(TemporaryDirectory().name)
    # --- sanity: mapping must not collapse to a constant -------------------
    if np.ptp(iso.predict(mu_vals)) < 1e-4:
        raise RuntimeError(
                        "Isotonic calibration collapsed to a flat line – check label mix!"
            )

    print(out_dir)
    f = out_dir / "iso_calibrator.pkl"
    print(f)
    joblib.dump(iso, f)
    return iso, f'''

def calibrate_isotonic(mu_vals, labels, out_dir: Path | None = None, *, y_min=0.0, y_max=1.0):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=y_min, y_max=y_max)
    iso.fit(mu_vals.astype(float), labels.astype(float))

    # handle None *before* mkdir
    if out_dir is None:
        out_dir = Path(TemporaryDirectory().name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # sanity check stays the same
    if np.ptp(iso.predict(mu_vals)) < 1e-4:
        raise RuntimeError("Isotonic calibration collapsed to a flat line – check label mix!")

    f = out_dir / "iso_calibrator.pkl"
    joblib.dump(iso, f)
    return iso, f



# --------------------------------------------------------------------- #
# Inference helper                                                      #
# --------------------------------------------------------------------- #
def load_calibrator(artefact_dir: Path) -> IsotonicRegression:
    """Return the previously‑saved isotonic regressor."""
    return joblib.load(artefact_dir / "iso_calibrator.pkl")


# === Unified µ→p mapping =====================================================

from typing import Iterable, Optional, Union

'''def map_mu_to_prob(
    mu: Union[float, np.ndarray, Iterable[float]],
    *,
    calibrator: Optional[IsotonicRegression] = None,
    artefact_dir: Optional[Path] = None,
    default_if_missing: float = None,
) -> np.ndarray:
    # sanitize to 1-D float array
    mu_arr = np.asarray(mu, dtype=float).reshape(-1)

    # try to load a calibrator if not provided
    iso = calibrator
    if iso is None and artefact_dir is not None:
        try:
            iso = load_calibrator(artefact_dir)
        except Exception:
            iso = None

    # if we have a calibrator, predict on finite values only
    if iso is not None:
        out = np.empty_like(mu_arr, dtype=float)
        mask = np.isfinite(mu_arr)
        if mask.any():
            try:
                out[mask] = iso.predict(mu_arr[mask])                  # 1-D path
            except ValueError:
                out[mask] = iso.predict(mu_arr[mask].reshape(-1, 1))   # 2-D (n,1) fallback
        if (~mask).any():
            out[~mask] = 0.5  # neutral prob for NaNs/infs
        return out

    # no calibrator → heuristic or constant default
    if default_if_missing is not None:
        return np.full_like(mu_arr, float(default_if_missing), dtype=float)

    # stable logistic-ish fallback (rank-preserving, non-explosive)
    scale = max(1e-3, np.median(np.abs(mu_arr[np.isfinite(mu_arr)])) * 5.0) if np.isfinite(mu_arr).any() else 1.0
    return 0.5 * (1.0 + np.tanh(mu_arr / scale))'''

# prediction_engine/calibration.py
#import numpy as np

'''def map_mu_to_prob(mu, meta):
    mu_arr = np.asarray(mu)
    if mu_arr.ndim == 1:
        pass
    elif mu_arr.ndim == 2 and mu_arr.shape[1] == 1:
        mu_arr = mu_arr.ravel()
    else:
        raise ValueError(f"[map_mu_to_prob] Unexpected mu shape {mu_arr.shape}")

    if "isotonic_model" in meta:
        model = meta["isotonic_model"]
        p = model.predict(mu_arr)
    else:
        p = mu_arr  # identity mapping

    if len(p) != len(mu_arr):
        raise ValueError(f"[map_mu_to_prob] Length mismatch: p={len(p)}, mu={len(mu_arr)}")

    return np.asarray(p)'''


def map_mu_to_prob(
    mu: Union[float, np.ndarray, Iterable[float]],
    *,
    calibrator: Optional[IsotonicRegression] = None,
    artefact_dir: Optional[Path] = None,
    default_if_missing: Optional[float] = None,
    meta: Optional[dict] = None,  # <-- tolerated for forward/back compat
) -> np.ndarray:
    """
    Unified µ -> probability (or expected return) mapper.

    Accepts either:
      - the legacy keyword form (calibrator=..., artefact_dir=..., default_if_missing=...)
      - or a temporary caller that passed (meta=...) holding a model
    """
    # ---- sanitize mu to 1D float array -----------------------------------
    mu_arr = np.asarray(mu, dtype=float)
    if mu_arr.ndim == 2 and mu_arr.shape[1] == 1:
        mu_arr = mu_arr.ravel()
    elif mu_arr.ndim > 2:
        raise ValueError(f"[map_mu_to_prob] Unexpected mu shape {mu_arr.shape}")
    # ensure 1-D
    mu_arr = mu_arr.reshape(-1)

    # ---- pick a calibrator ------------------------------------------------
    iso = calibrator
    # allow meta-based injection (from your experimental version)
    if iso is None and isinstance(meta, dict) and "isotonic_model" in meta:
        iso = meta["isotonic_model"]

    # lazy-load from disk if needed
    if iso is None and artefact_dir is not None:
        try:
            iso = load_calibrator(Path(artefact_dir))
        except Exception:
            iso = None

    # ---- map µ -> p (or expected value) ----------------------------------
    if iso is not None:
        out = np.empty_like(mu_arr, dtype=float)
        mask = np.isfinite(mu_arr)
        if mask.any():
            try:
                out[mask] = iso.predict(mu_arr[mask])               # 1-D path
            except ValueError:
                out[mask] = iso.predict(mu_arr[mask].reshape(-1, 1))  # (n,1) fallback
        if (~mask).any():
            out[~mask] = 0.5  # neutral for NaNs/infs
        return out

    # ---- no calibrator available -----------------------------------------
    if default_if_missing is not None:
        return np.full_like(mu_arr, float(default_if_missing), dtype=float)

    # stable monotone fallback (keeps rank, prevents explosions)
    finite = np.isfinite(mu_arr)
    if finite.any():
        scale = max(1e-3, np.median(np.abs(mu_arr[finite])) * 5.0)
    else:
        scale = 1.0
    return 0.5 * (1.0 + np.tanh(mu_arr / scale))

