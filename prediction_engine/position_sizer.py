# --------------------------------------------------------------------
# prediction_engine/position_sizer.py          – Kelly & liquidity cap
# --------------------------------------------------------------------
from __future__ import annotations
import numpy as np

class KellySizer:
    """
    Convert an EVEngine result into a position size ∈ [0, 1].

    • Raw Kelly  :  f* = μ / σ²↓   (downside-variance Kelly)
    • Convex clip:  f_clip = min(f*, f_max)
    • Liquidity  :  cap by %ADV   (linear taper after 10 % ADV)

    All sizing is *fraction of notional capital* (1 = 100 %).
    """

    def __init__(self,
                 f_max: float = 0.40,
                 adv_cap_pct: float | None = None) -> None:
        self.f_max      = float(f_max)
        #self.adv_cap_pct = float(adv_cap_pct)
        # allow YAML override; fall back to config defaults
        from scanner.config import load as _load_cfg
        cfg_liq = _load_cfg().get("liquidity", {})
        adv_cap_pct = adv_cap_pct if adv_cap_pct is not None else cfg_liq.get("adv_cap_pct", 20.0)
        self.adv_cap_pct = float(adv_cap_pct)
    # ----------------------------------------------------------------
    def size(self, mu: float, var_down: float, adv_percentile: float | None) -> float:
        """Return recommended position size ∈ [0, f_max]."""
        #if mu <= 0.0 or var_down <= 0.0:
        #    return 0.0

        # allow both long (µ>0) and short (µ<0)
        if var_down <= 0.0:
            return 0.0

        f_raw = mu / (var_down + 1e-12)
        #f_raw = max(f_raw, 0.0)

        # clip raw fraction to [-f_max, +f_max]
        f_raw = float(np.sign(f_raw) * min(abs(f_raw), self.f_max))

        # Hard Kelly clip
        f_kelly = min(f_raw, self.f_max)

        # Liquidity taper – LOG curve: 100 % size ≤ 10 % ADV,
        # then decays as   scale = 1 − log1p(adv − 10) / log1p(cap − 10)
        if adv_percentile is not None and adv_percentile > 10.0:
            adv = min(adv_percentile, self.adv_cap_pct)
            decay = np.log1p(adv - 10.0) / np.log1p(self.adv_cap_pct - 10.0)
            scale = max(0.0, 1.0 - decay)
            f_kelly *= scale

        return float(np.clip(f_kelly, 0.0, self.f_max))
