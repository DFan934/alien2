from pathlib import Path

def test_broker_bad_import_removed():
    p = Path("execution/broker_api.py")
    assert p.exists(), "execution/broker_api.py not found"
    src = p.read_text()
    assert "from core.contracts" not in src, "Remove 'from core.contracts' import here"
