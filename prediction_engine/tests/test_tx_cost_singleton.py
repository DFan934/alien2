import numpy as np
from prediction_engine.tx_cost import BasicCostModel
from prediction_engine import tx_cost

def test_legacy_estimate_forwards_to_singleton():
    m = BasicCostModel()
    # try a few adv percentiles; should match singleton wrapper exactly
    for adv in [None, 0, 5, 10, 20]:
        a = m.estimate(half_spread=None, adv_percentile=adv)
        b = tx_cost.estimate(half_spread=None, adv_percentile=adv)
        assert np.isclose(a, b)
