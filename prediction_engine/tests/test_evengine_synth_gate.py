import numpy as np
from prediction_engine.ev_engine import EVEngine

def test_distance_gate_triggers_synth(monkeypatch):
    rng = np.random.default_rng(0)
    # far-away centroids so dist² > τ
    centers = rng.normal(loc=10.0, scale=1.0, size=(4, 5)).astype(np.float32)
    engine = EVEngine(
        centers=centers,
        mu=np.zeros(4, dtype=np.float32),
        var=np.ones(4, dtype=np.float32),
        var_down=np.ones(4, dtype=np.float32),
        h=0.5,
        tau_dist=1.0,  # small τ ⇒ gate will fire
    )

    calls = {"n": 0}
    # Save the original function that we are going to patch
    original_synth_cache = engine._synth_cache

    def spy_cache_call(*args, **kwargs):
        """Our spy that intercepts the call."""
        calls["n"] += 1
        # Now, continue to the original function
        return original_synth_cache(*args, **kwargs)

    # Use monkeypatch to replace _synth_cache with our spy
    monkeypatch.setattr(engine, "_synth_cache", spy_cache_call)

    engine.evaluate(rng.normal(size=5).astype(np.float32))
    assert calls["n"] == 1  # synthesiser was called