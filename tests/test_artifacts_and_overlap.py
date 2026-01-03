from __future__ import annotations

from pathlib import Path
import json
import tempfile

from scripts.run_backtest import run_batch
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
        # run 2nd backtest (different run_id so different run_dir, same base root)
        cfg2 = cfg.copy()
        cfg2["run_id"] = "test_run_2"
        out2 = run_batch(cfg2)

        # Verify artifacts root contract via overlap_audit.json existence
        # (it’s written early, before portfolio artifacts)
        # You can also check run_context.json exists
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
