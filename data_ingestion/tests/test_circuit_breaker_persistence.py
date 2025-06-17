import json
import time

def test_circuit_breaker_persistence(ingestion_manager, tmp_state_file):
    # force a breaker trip
    ingestion_manager._trip_breaker("tradier")
    first_trip = ingestion_manager.breaker_state["tradier"]["last_trip"]

    # simulate a "restart"
    new_mgr = ingestion_manager.__class__(state_path=tmp_state_file)
    assert new_mgr.breaker_state["tradier"]["last_trip"] == first_trip

    # breaker window still active
    assert not new_mgr._breaker_window_expired("tradier")
