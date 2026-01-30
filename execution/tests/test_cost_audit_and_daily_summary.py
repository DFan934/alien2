# execution/tests/test_phase13_cost_audit_and_daily_summary.py
from __future__ import annotations

from pathlib import Path

import pandas as pd

from execution.live_daily_summary import generate_daily_summary


def test_phase13_daily_summary_written_even_with_no_trades(tmp_path: Path):
    # No trades, no artifacts present -> must still write outputs
    out = generate_daily_summary(tmp_path, cfg={"equity": 100_000.0}, default_equity=100_000.0)

    assert (tmp_path / "daily_summary.json").exists()
    assert (tmp_path / "cost_audit.json").exists()
    assert (tmp_path / "equity_curve.parquet").exists()

    curve = pd.read_parquet(tmp_path / "equity_curve.parquet")
    assert len(curve) >= 1
    assert set(curve.columns) >= {"ts_utc", "equity"}

    assert "counts" in out
    assert out["counts"]["attempted_actions"] == 0
    assert out["counts"]["orders_submitted"] == 0
    assert out["counts"]["fills"] == 0
