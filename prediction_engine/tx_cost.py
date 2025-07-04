# ---------------------------------------------------------------------------
# tx_cost.py
# ---------------------------------------------------------------------------
"""Simple square‑root‑impact slippage model (unit‑tested elsewhere)."""

import numpy as np

# calibrated nightly via scripts/nightly_calibrate.py
_IMPACT_COEFF = 0.9e-4


def estimate(order_size: float, ADV: float) -> float:  # noqa: D401
    """Return per‑share slippage given order size and daily vol (ADV)."""
    frac = np.clip(order_size / ADV, 1e-6, 1)
    return _IMPACT_COEFF * np.sqrt(frac)