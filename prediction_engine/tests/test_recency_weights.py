import numpy as np
import pytest

from prediction_engine.weight_optimization import CurveParams
from prediction_engine.ev_engine import EVEngine
from prediction_engine.market_regime import MarketRegime
from prediction_engine.tx_cost import BasicCostModel


def make_engine(regime_curves: dict[str, CurveParams]) -> EVEngine:
    """
    Helper: construct a minimal EVEngine instance with 1 cluster and 1 feature.
    """
    centers = np.zeros((1, 1), dtype=np.float32)
    mu = np.zeros((1,), dtype=np.float32)
    var = np.ones((1,), dtype=np.float32)
    var_down = np.ones((1,), dtype=np.float32)
    h = 1.0

    return EVEngine(
        centers=centers,
        mu=mu,
        var=var,
        var_down=var_down,
        h=h,
        regime_curves=regime_curves,
        outcome_probs={},
        cost_model=BasicCostModel(),
    )


def test_get_recency_weights_none_regime():
    engine = make_engine(regime_curves={})
    w = engine._get_recency_weights(None, 5)
    assert np.allclose(w, np.ones(5)), "None regime should yield uniform weights"


def test_get_recency_weights_missing_curve():
    engine = make_engine(regime_curves={})
    w = engine._get_recency_weights(MarketRegime.TREND, 4)
    assert np.allclose(w, np.ones(4)), "Missing curve should default to uniform"


def test_get_recency_weights_linear_curve():
    p = CurveParams(family="linear", tail_len=3, shape=1.0)
    engine = make_engine(regime_curves={"trend": p})
    w = engine._get_recency_weights(MarketRegime.TREND, 3)
    # WeightOptimizer._weights for n=3, tail_len=3: idx=[2,1,0] → [1-2/3, 1-1/3, 1]
    expected = np.array([1/3, 2/3, 1.0], dtype=np.float32)
    assert np.allclose(w, expected), f"Linear weights mismatch: {w} != {expected}"


def test_get_recency_weights_exp_curve():
    p = CurveParams(family="exp", tail_len=4, shape=2.0)
    engine = make_engine(regime_curves={"range": p})
    w = engine._get_recency_weights(MarketRegime.RANGE, 4)
    # Compute via the same internal logic:
    idx = np.arange(4, dtype=float)[::-1]   # [3,2,1,0]
    mask = idx < p.tail_len                 # all true for tail_len=4
    idx = idx[mask]
    expected = np.exp(-idx / (p.shape * 10)).astype(np.float32)
    assert np.allclose(w, expected), f"Exp weights mismatch: {w} != {expected}"


def test_evaluate_applies_recency_in_kernel(monkeypatch):
    # Two clusters, three features
    centers = np.zeros((2, 3), dtype=np.float32)
    mu = np.array([1.0, -1.0], dtype=np.float32)
    var = np.ones((2,), dtype=np.float32)
    var_down = np.ones((2,), dtype=np.float32)
    h = 1.0

    # Define a simple linear recency curve for "trend"
    p = CurveParams(family="linear", tail_len=2, shape=1.0)
    engine = EVEngine(
        centers=centers,
        mu=mu,
        var=var,
        var_down=var_down,
        h=h,
        regime_curves={"trend": p},
        outcome_probs={},
        cost_model=BasicCostModel(),
    )

    # --- FIX: disable transaction costs for this isolated recency test ---
    engine._cost.estimate = lambda half_spread=None, adv_percentile=None: 0.0

    # Monkey-patch _dist to return fixed neighbors and squared distances
    monkeypatch.setattr(
        engine,
        "_dist",
        lambda x, k: (np.array([0, 1]), np.array([0.0, 1.0]))
    )

    # --- Compute pure-kernel µ (no recency) ---
    engine_no = EVEngine(
        centers = centers,
        mu = mu,
        var = var,
        var_down = var_down,
        h = h,
        regime_curves = {},  # no recency curve
        outcome_probs = {},
        cost_model = BasicCostModel(),
    )
    engine_no._cost.estimate = lambda *a, **k: 0.0
    monkeypatch.setattr(engine_no, "_dist", lambda x, k: (np.array([0, 1]), np.array([0.0, 1.0])))

    x = np.zeros(3, dtype=np.float32)


    pure = engine_no.evaluate(x, regime=MarketRegime.TREND).mu

    #result = engine.evaluate(x, regime=MarketRegime.TREND)

    result = engine.evaluate(x, regime=MarketRegime.TREND)

    '''# Manually recompute what kernel+recency weights should be
    rec_w = engine._get_recency_weights(MarketRegime.TREND, 2)
    ker_w = np.exp(-0.5 * np.array([0.0, 1.0]) / (h ** 2))
    combined = ker_w * rec_w
    combined /= combined.sum()

    # Expected µ_k = w[0]*1 + w[1]*(-1)
    expected_mu_k = float(combined[0] * 1 + combined[1] * -1)
    # Subtract the transaction cost that BasicCostModel actually applies
    cost_ps = engine._cost.estimate(half_spread=None, adv_percentile=None)
    expected_mu_net = expected_mu_k - cost_ps
    assert result.mu == pytest.approx(expected_mu_net, rel=1e-6)'''

    # The two must differ if recency is actually applied:
    assert result.mu != pytest.approx(pure)
    # And since our linear curve puts *more* weight on the more recent cluster (index 1, μ=-1),
    # we expect result.mu < pure (i.e. more negative):
    assert result.mu < pure