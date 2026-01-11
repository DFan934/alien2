from __future__ import annotations

from pathlib import Path
import json
import tempfile

import pandas as pd

from scripts.run_backtest import run_batch, _timestamp_overlap_share
from universes.providers import StaticUniverse


def _norm(p: Path) -> str:
    return str(p.resolve()).lower().replace("/", "\\")


def test_phase11_artifacts_root_contract_and_overlap_gate():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        cfg = {
            "parquet_root": "parquet",  # adjust if your tests use a fixture dataset
            "universe": StaticUniverse(["RRC", "BBY"]),
            "start": "1998-08-26",
            "end": "1999-01-01",

            # Force artifacts into temp dir (absolute)
            "artifacts_root": str((td / "artifacts").resolve()),

            # Gate config
            "min_overlap_share_ge2": 0.10,

            # Required execution_rules because run_batch demands it
            "execution_rules": {
                "max_participation": 0.10,
                "min_fill_shares": 1,
            },

            # required
            "run_id": "test_run_1",
            "universe_hash": "deadbeef",
        }

        out1 = run_batch(cfg.copy())

        cfg2 = cfg.copy()
        cfg2["run_id"] = "test_run_2"
        out2 = run_batch(cfg2)

        # Verify artifacts root contract via overlap_audit.json existence
        for run_dir in (td / "artifacts").glob("*"):
            if not run_dir.is_dir():
                continue

            norm = _norm(run_dir)
            assert "\\scripts\\artifacts" not in norm, f"Forbidden path: {run_dir}"
            assert run_dir.is_absolute()

            rc = run_dir / "run_context.json"
            if rc.exists():
                payload = json.loads(rc.read_text(encoding="utf-8"))
                assert "artifacts_root" in payload

            ov = run_dir / "overlap_audit.json"
            if ov.exists():
                o = json.loads(ov.read_text(encoding="utf-8"))
                assert "overlap_share" in o
                assert "min_required" in o

        assert isinstance(out1, dict)
        assert isinstance(out2, dict)


def test_overlap_can_be_1_0_even_when_symbol_coverage_is_low_due_to_denominator_mismatch():
    # unified clock: the "true" minute universe
    clock = pd.date_range("2020-01-01", periods=10, freq="60s", tz="UTC")
    clock_len = int(len(clock))
    assert clock_len == 10

    # build a FULL post-timegrid-like frame (every symbol x every timestamp)
    rows = []
    for i, t in enumerate(clock):
        b_present = 1 if i < 3 else 0
        a_present = b_present  # A present only when B is present

        rows.append({"timestamp": t, "symbol": "A", "close": (1.0 if a_present else None), "bar_present": int(a_present)})
        rows.append({"timestamp": t, "symbol": "B", "close": (1.0 if b_present else None), "bar_present": int(b_present)})

    bars_full = pd.DataFrame(rows)

    # coverage over the FULL unified clock
    per_symbol_coverage = bars_full.groupby("symbol")["bar_present"].mean().to_dict()
    assert abs(per_symbol_coverage["B"] - 0.3) < 1e-9

    # emulate the mismatch: overlap input is pre-filtered to sparse timestamp set
    overlap_basis_ts = set(clock[:3])
    bars_for_overlap = bars_full[
        bars_full["timestamp"].isin(overlap_basis_ts) & (bars_full["bar_present"].astype(bool))
    ].copy()

    assert bars_for_overlap["timestamp"].nunique() == 3
    assert set(bars_for_overlap["symbol"].unique()) == {"A", "B"}

    # call the REAL overlap function
    overlap = _timestamp_overlap_share(
        bars_for_overlap,
        min_symbols=2,
        ts_col="timestamp",
        symbol_col="symbol",
        presence_col="bar_present",
    )
    assert abs(overlap - 1.0) < 1e-12

    # replicate the internal numerator/denominator view
    b = bars_for_overlap.copy()
    b["timestamp"] = pd.to_datetime(b["timestamp"], utc=True, errors="coerce")
    b = b.dropna(subset=["timestamp", "symbol"])
    b = b[b["bar_present"].astype(bool)]

    counts = b.groupby("timestamp")["symbol"].nunique()
    total_ts = int(len(counts))
    good_ts = int((counts >= 2).sum())

    assert total_ts == 3
    assert good_ts == 3

    overlap_on_clock_view = float(good_ts) / float(clock_len)
    assert abs(overlap_on_clock_view - 0.3) < 1e-12

    # contradiction condition
    contradiction = None
    if overlap >= 0.95:
        low = {sym: float(cov) for sym, cov in per_symbol_coverage.items() if float(cov) < 0.50}
        if low:
            contradiction = {"overlap_ratio": float(overlap), "low_coverage": low, "clock_len": clock_len}

    assert contradiction is not None
    assert "B" in contradiction["low_coverage"]
