import numpy as np
from prediction_engine.position_sizer import KellySizer

def test_kelly_basic():
    ks = KellySizer(f_max=0.4)
    #  μ = 0.02 ,  σ²↓ = 0.04   →  f* = 0.5  → clipped to 0.4
    s = ks.size(mu=0.02, var_down=0.04, adv_percentile=5.0)
    assert abs(s - 0.4) < 1e-6

def test_liquidity_taper():
    ks = KellySizer(f_max=0.4, adv_cap_pct=20.0)
    # Same μ/σ but high liquidity percentile (15 %) → 50 % taper
    s = ks.size(mu=0.02, var_down=0.04, adv_percentile=15.0)
    # raw clip 0.4 then ×0.5 = 0.2
    assert abs(s - 0.2) < 1e-6

def test_negative_mu_gives_zero():
    ks = KellySizer()
    assert ks.size(mu=-0.01, var_down=0.05, adv_percentile=5.0) == 0.0
