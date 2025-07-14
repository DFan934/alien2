from execution.stop_manager import StopManager
from execution.position_store import PositionStore

def test_stop_manager_profile_override():
    store = PositionStore(":memory:")
    sm = StopManager(store)
    sm.register_position("XYZ", entry_atr=1.0)
    store.add_position("sig1", "XYZ", "BUY", 100, 100.0, 98.5, 104.0)
    prof = {"atr_mult": 2.0}
    new = sm.update("XYZ", "BUY", 100.0, -0.003, 0.0, 1.0, profile=prof)
    assert new == 100.0 - 2.0 * 1.0
