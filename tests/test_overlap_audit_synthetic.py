from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

#from feature_engineering.utils.timegrid import _timestamp_overlap_share
from scripts.run_backtest import _timestamp_overlap_share

def _build_and_write_overlap_audit(tmp_artifacts_root: Path, bars: pd.DataFrame, clock_len: int, overlap: float) -> dict:
    # Per-symbol coverage/counts (mirror run_backtest logic)
    per_symbol_counts = {}
    per_symbol_coverage = {}

    g = bars.groupby("symbol", sort=False)["bar_present"]
    # IMPORTANT: treat NaN as 0 for coverage (so coverage is honest)
    per_symbol_coverage = pd.to_numeric(g.mean(), errors="coerce").fillna(0).astype(float).to_dict()

    for sym, s in g:
        vc = s.value_counts(dropna=False).to_dict()
        norm_counts = {}
        for k, v in vc.items():
            if k is None or (isinstance(k, float) and pd.isna(k)):
                key = "NaN"
            else:
                try:
                    key = str(int(k))
                except Exception:
                    key = str(k)
            norm_counts[key] = int(v)
        per_symbol_counts[str(sym)] = norm_counts

    # compute overlap integers the same way overlap function does internally
    b = bars.copy()
    b["timestamp"] = pd.to_datetime(b["timestamp"], utc=True, errors="coerce")
    b = b.dropna(subset=["timestamp", "symbol"])

    bp = pd.to_numeric(b["bar_present"], errors="coerce").fillna(0)
    b = b[bp > 0]

    counts = b.groupby("timestamp")["symbol"].nunique()
    total_ts = int(len(counts))
    good_ts = int((counts >= 2).sum())

    contradiction = None
    low = {sym: float(cov) for sym, cov in per_symbol_coverage.items() if float(cov) < 0.50}
    if overlap >= 0.95 and low:
        contradiction = {"overlap_ratio": float(overlap), "low_coverage": low, "clock_len": int(clock_len),
                         "good_ts": good_ts, "total_ts": total_ts}

    payload = {
        "basis": "bars_after_timegrid",
        "ts_col": "timestamp",
        "symbol_col": "symbol",
        "presence_col": "bar_present",
        "min_symbols": 2,

        "n_unique_ts_total": total_ts,
        "n_ts_meeting_min_symbols": good_ts,
        "overlap_ratio": float(overlap),

        "clock_len": int(clock_len),
        "overlap_on_clock_view": (float(good_ts) / float(clock_len)) if clock_len else None,

        "per_symbol_coverage": per_symbol_coverage,
        "per_symbol_counts": per_symbol_counts,

        "contradiction": contradiction,
    }

    diag_dir = tmp_artifacts_root / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    (diag_dir / "overlap_audit.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def test_overlap_audit_required_fields_and_realistic_overlap(tmp_path: Path):
    # 10-minute clock
    ts = pd.date_range("2020-01-01 09:30:00", periods=10, freq="T", tz="UTC")

    # Symbol A present all 10
    a = pd.DataFrame({"timestamp": ts, "symbol": "AAA", "bar_present": 1})

    # Symbol B present only on 2 timestamps (sparse)
    b = pd.DataFrame({"timestamp": ts, "symbol": "BBB", "bar_present": [1, 0, 0, 0, 0, 0, 0, 0, 0, 1]})

    bars = pd.concat([a, b], ignore_index=True)

    # Real overlap from the true function
    overlap = _timestamp_overlap_share(
        bars,
        min_symbols=2,
        ts_col="timestamp",
        symbol_col="symbol",
        presence_col="bar_present",
    )

    payload = _build_and_write_overlap_audit(tmp_path, bars, clock_len=len(ts), overlap=overlap)

    # --- Required fields
    for k in [
        "n_unique_ts_total",
        "n_ts_meeting_min_symbols",
        "overlap_ratio",
        "per_symbol_coverage",
        "per_symbol_counts",
    ]:
        assert k in payload, f"missing required key: {k}"

    # --- Coverage proves BBB is sparse
    assert payload["per_symbol_coverage"]["BBB"] < 0.50

    # --- Overlap should NOT be inflated to ~1.0 in this sparse scenario
    assert payload["overlap_ratio"] < 0.95


def test_overlap_contradiction_log_path_is_exercised(tmp_path: Path, monkeypatch, capsys):
    # Build a scenario with low coverage but then force overlap high via monkeypatch
    ts = pd.date_range("2020-01-01 09:30:00", periods=10, freq="T", tz="UTC")
    a = pd.DataFrame({"timestamp": ts, "symbol": "AAA", "bar_present": 1})
    b = pd.DataFrame({"timestamp": ts, "symbol": "BBB", "bar_present": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]})
    bars = pd.concat([a, b], ignore_index=True)

    # Force overlap high to ensure contradiction branch triggers
    forced_overlap = 0.999
    payload = _build_and_write_overlap_audit(tmp_path, bars, clock_len=len(ts), overlap=forced_overlap)

    # contradiction must exist in payload in this forced condition
    assert payload["contradiction"] is not None
    assert "low_coverage" in payload["contradiction"]
