import numpy as np, pandas as pd
from prediction_engine.market_regime import RegimeDetector, MarketRegime

def test_regime_detector_labels():
    td = RegimeDetector()

    # ---- synthetic trend (sinusoid with drift) ---------------------
    t = np.linspace(0, 10, 300)
    price = 0.1 * t + np.sin(t)        # clear direction
    bars  = pd.DataFrame({"close": price,
                          "high":  price + 0.01,
                          "low":   price - 0.01})
    regime = td.update(bars)
    assert regime is MarketRegime.TREND

    # ---- high-vol random walk  -------------------------------------
    rnd = np.cumsum(np.random.normal(0, 1, 300))
    bars2 = pd.DataFrame({"close": rnd,
                          "high":  rnd + 0.05,
                          "low":   rnd - 0.05})
    regime2 = td.update(bars2)
    assert regime2 is MarketRegime.VOLATILE
