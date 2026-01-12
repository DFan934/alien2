import json
from pathlib import Path

import pandas as pd
import pytest

from feature_engineering.utils.consistency_gate import enforce_portfolio_bar_coverage_gate


def _make_symbol_frame(sym: str, clock: pd.DatetimeIndex, present_mask) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": clock,
        "symbol": [sym] * len(clock),
        "bar_present": present_mask,
    })


def test_coverage_gate_writes_artifact_and_detects_failing_symbol_abort(tmp_path: Path):
    clock = pd.date_range("1999-01-04 14:30:00", periods=100, freq="60s", tz="UTC")

    a = _make_symbol_frame("AAA", clock, [1] * len(clock))
    b = _make_symbol_frame("BBB", clock, [1] * 20 + [0] * 80)  # 0.20 coverage

    bars_std = pd.concat([a, b], ignore_index=True)

    res = enforce_portfolio_bar_coverage_gate(
        bars_std,
        artifacts_root=tmp_path,
        threshold=0.75,
        mode="abort",
        symbol_col="symbol",
        presence_col="bar_present",
        clock_len=len(clock),
    )

    # Artifact exists and has required fields
    p = tmp_path / "diagnostics" / "coverage_gate.json"
    assert p.exists()
    payload = json.loads(p.read_text())

    assert payload["threshold"] == 0.75
    assert payload["mode"] == "abort"
    assert payload["presence_col"] == "bar_present"
    assert payload["clock_len"] == len(clock)
    assert payload["per_symbol_coverage"]["AAA"] == 1.0
    assert payload["per_symbol_coverage"]["BBB"] == pytest.approx(0.20, rel=1e-6)
    assert payload["failing_symbols"] == ["BBB"]
    assert "BBB" not in payload["kept_symbols"]


def test_coverage_gate_drop_mode_writes_dropped_symbols(tmp_path: Path):
    clock = pd.date_range("1999-01-04 14:30:00", periods=50, freq="60s", tz="UTC")

    a = _make_symbol_frame("AAA", clock, [1] * len(clock))
    b = _make_symbol_frame("BBB", clock, [0] * len(clock))  # 0.0 coverage

    bars_std = pd.concat([a, b], ignore_index=True)

    res = enforce_portfolio_bar_coverage_gate(
        bars_std,
        artifacts_root=tmp_path,
        threshold=0.75,
        mode="drop",
        symbol_col="symbol",
        presence_col="bar_present",
        clock_len=len(clock),
    )

    assert res.failing_symbols == ["BBB"]
    assert res.kept_symbols == ["AAA"]

    dropped = tmp_path / "diagnostics" / "dropped_symbols.json"
    assert dropped.exists()
    d = json.loads(dropped.read_text())
    assert d["dropped"] == ["BBB"]
    assert d["kept"] == ["AAA"]
