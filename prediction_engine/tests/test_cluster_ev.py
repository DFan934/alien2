# tests/test_cluster_ev.py

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from prediction_engine.ev_engine import EVEngine
from prediction_engine.tx_cost import BasicCostModel

ARTIFACT_DIR = Path(__file__).parents[2] / "weights"

@pytest.fixture(scope="module")
def ev_cost_free():
    """An EVEngine with zero transaction cost."""
    cost = BasicCostModel()
    return EVEngine.from_artifacts(ARTIFACT_DIR, cost_model=cost)

@pytest.fixture(scope="module")
def ev_cost_aware():
    """An EVEngine with built‑in cost model (default)."""
    return EVEngine.from_artifacts(ARTIFACT_DIR, cost_model=BasicCostModel())

@pytest.fixture(scope="module")
def cluster_centers():
    return np.load(ARTIFACT_DIR / "centers.npy")

@pytest.fixture(scope="module")
def cluster_stats():
    data = np.load(ARTIFACT_DIR / "cluster_stats.npz", allow_pickle=True)
    return data["mu"]

def test_centroid_mu_fraction(ev_cost_free, cluster_centers):
    """
    Sanity‑check: at least one of the first 20 centroids should be positive EV
    when we ignore costs (half_spread=0).
    """
    mus = []
    for center in cluster_centers[:20]:
        evr = ev_cost_free.evaluate(center, half_spread=0)
        mus.append(evr.mu)
    share_pos = np.mean(np.array(mus) > 0)
    assert share_pos > 0, (
        "All of the first 20 centroids had μ ≤ 0, "
        "but we'd expect at least one cost‑free positive cluster."
    )

def test_cluster_stats_summary(cluster_stats):
    """
    Verify that the precomputed cluster_stats.npz actually contains
    a reasonable spread of positive μ clusters.
    """
    series = pd.Series(cluster_stats)
    frac_positive = (series > 0).mean()
    # since we know from earlier that ~56% of clusters are positive,
    # we assert a loose bound to catch regressions.
    assert 0.4 < frac_positive < 0.8, (
        f"Unexpected positive‑μ fraction: {frac_positive:.2%}. "
        "Check your artifact build."
    )

@pytest.mark.parametrize("idx", [0, 1, 2, 3, 4])
def test_individual_cluster_info(ev_cost_aware, cluster_centers, idx):
    """
    Smoke‑test that each of the first 5 centroids can be evaluated
    under the default (cost‑aware) EVEngine without errors, and returns
    a finite μ.
    """
    center = cluster_centers[idx]
    evr = ev_cost_aware.evaluate(center)
    assert np.isfinite(evr.mu), f"Cluster #{idx} produced non‑finite μ"

