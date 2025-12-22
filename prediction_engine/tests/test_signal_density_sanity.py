import os
from pathlib import Path
import pandas as pd

def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur, *cur.parents]:
        if (p / "scripts" / "run_backtest.py").exists():
            return p
        if (p / ".git").exists():
            return p
    return cur.parents[0]

def _latest_run(artifacts_root: Path) -> Path:
    runs = sorted([p for p in artifacts_root.glob("a2_*") if p.is_dir()])
    assert runs, f"No a2_* runs under {artifacts_root}"
    return runs[-1]

def test_latest_artifact_signal_density_and_schema():
    repo = _find_repo_root(Path(__file__))
    artifacts_root = Path(os.environ.get("ARTIFACTS_ROOT", repo / "artifacts")).resolve()

    run = _latest_run(artifacts_root)
    portfolio = run / "portfolio"

    trades_path = portfolio / "trades.csv"
    decisions_path = portfolio / "decisions.csv"

    assert trades_path.exists(), f"Missing {trades_path}"
    assert decisions_path.exists(), f"Missing {decisions_path}"

    trades = pd.read_csv(trades_path)
    decisions = pd.read_csv(decisions_path)

    # 1) schema sanity
    assert "symbol" in trades.columns, "trades.csv must include 'symbol'"
    assert "symbol" in decisions.columns, "decisions.csv must include 'symbol'"

    # 2) density sanity (tune these as your Phase-3 target)
    assert len(trades) >= 300, f"Too few trades: {len(trades)}"
    assert len(decisions) >= 2000, f"Too few decisions rows: {len(decisions)}"

    # 3) probability sanity
    p_cols = [c for c in trades.columns if c in ("p", "p_cal", "p_final")]
    assert p_cols, f"Expected a probability column in trades.csv, found none. cols={list(trades.columns)}"
