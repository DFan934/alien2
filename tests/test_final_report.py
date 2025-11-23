import json
from pathlib import Path
import pandas as pd
import numpy as np

from scripts.a2_report import generate_final_report, GateThresholds

def _mk_fold(root: Path, k: int, with_slippage: bool = False):
    fd = root / f"fold_{k:03d}"; (fd / "metrics").mkdir(parents=True, exist_ok=True)
    # minimal fold_metrics payload
    fm = {
        "n_decisions": 100,
        "n_trades_matched": 80,
        "ece": 0.02,              # <= 3%
        "brier": 0.19,            # arbitrary
        "deciles": 10,
        "decile_table": [],
        "fallback_rate": 0.10,    # <= 15%
        "analog_fidelity_by_bucket": {
            "TREND": {"long": 0.35, "short": 0.31},
            "RANGE": {"long": 0.32},
            "VOL":   {"short": 0.33}
        },
        "residual_cusum_3sigma_breaches": 0,
        "psi": None
    }
    (fd / "metrics" / "fold_metrics.json").write_text(json.dumps(fm, indent=2), encoding="utf-8")

    # trades with simple returns
    n = 120
    rng = np.random.default_rng(0)
    tr = pd.DataFrame({
        "entry_ts": pd.date_range("2000-01-01", periods=n, freq="T"),
        "exit_ts":  pd.date_range("2000-01-01", periods=n, freq="T"),
        "qty": 1.0,
        "realized_pnl_after_costs": rng.normal(0.001, 0.01, size=n),
    })
    if with_slippage:
        tr["slippage_model_bps"] = 5.0
        tr["slippage_realized_bps"] = 7.0
    (fd / "trades.parquet").write_bytes(b"")  # create first so engine won't error if parquet engine missing
    tr.to_parquet(fd / "trades.parquet", index=False)

def test_generate_final_report(tmp_path: Path):
    exp = tmp_path / "expanding"; exp.mkdir(parents=True, exist_ok=True)
    rol = tmp_path / "rolling";   rol.mkdir(parents=True, exist_ok=True)
    _mk_fold(exp, 0, with_slippage=True)
    _mk_fold(exp, 1, with_slippage=True)
    _mk_fold(rol, 0, with_slippage=False)  # no slippage columns -> gates marked UNKNOWN (allowed)

    out = tmp_path / "out"; out.mkdir(parents=True, exist_ok=True)
    thr = GateThresholds()
    rep = generate_final_report(expanding_root=exp, rolling_root=rol, out_dir=out,
                                thresholds=thr, dev_brier_reference=0.18)

    # Files exist
    assert (out / "final_report.json").exists()
    assert (out / "final_report.html").exists()

    # PASS flags present and boolean
    data = json.loads((out / "final_report.json").read_text())
    assert isinstance(data["pass_flags"]["expanding"], bool)
    assert isinstance(data["pass_flags"]["rolling"], bool)

    # If expanding meets gates, promotion_decision should be True when rolling does not contradict badly
    assert "promotion_decision" in data
