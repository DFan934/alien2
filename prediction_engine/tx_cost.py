# ---------------------------------------------------------------------------
# tx_cost.py  – square-root impact with optional liquidity flag
# ---------------------------------------------------------------------------
import numpy as np

_IMPACT_COEFF = 9e-5    # nightly calibration

def estimate(
    order_size: float = 1.0,
    ADV: float = 1.0,
    *,                       # keyword-only from here on
    adv_percentile: float | None = None,
) -> float:
    """
    Return per-share slippage.

    Parameters
    ----------
    order_size
        Shares in the order (defaults to 1 for back-tests).
    ADV
        Average daily volume (shares).
    adv_percentile
        Liquidity percentile 0–100 – ignored for now but accepted so that the
        prediction engine can pass it without raising a TypeError.
    """
    frac = np.clip(order_size / ADV, 1e-6, 1.0)
    return _IMPACT_COEFF * np.sqrt(frac)
