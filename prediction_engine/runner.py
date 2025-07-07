# =============================================================================
# prediction_engine/runner.py – authoritative version (clean, single section)
# =============================================================================
"""One‑click smoke test for the prediction‑engine stack.

Run with:
    python -m prediction_engine.runner

• Hard‑pointed to the **feature_engineering/feature_parquet** folder, so it only
  sees feature‑engineered tables that contain numeric columns.
• Automatically builds cluster centroids & kernel bandwidth if the weights
  directory is empty (works even with tiny samples by shrinking *n_clusters*).
• Falls back to KNN‑only EVEngine if no RandomForest feature‑weight file is
  found.
"""

from __future__ import annotations
import asyncio, json, logging, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from prediction_engine.ev_engine import EVEngine
from prediction_engine.path_cluster_engine import PathClusterEngine
from prediction_engine.execution.manager import ExecutionManager
from prediction_engine.execution.risk_manager import RiskManager
from prediction_engine.testing_validation.backtester import (
    AsyncBacktester,
    BrokerStub,
)
from types import SimpleNamespace   # ← latency-monitor stub


logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("runner")

ROOT = Path(__file__).resolve().parents[1]
PARQUET_DIR = ROOT / "feature_engineering" / "feature_parquet"  # only this dir
WEIGHTS_DIR = ROOT / "weights"; WEIGHTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1) Discover parquet slice – pick largest file under feature_parquet
# ---------------------------------------------------------------------------

def find_parquet() -> Path:
    files = list(PARQUET_DIR.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No parquet under {PARQUET_DIR}. Run feature_engineering/run_pipeline.py"
        )
    return max(files, key=lambda p: p.stat().st_size)


# ---------------------------------------------------------------------------
# 2) Ensure cluster weights exist (cluster_stats & kernel_bandwidth)
# ---------------------------------------------------------------------------

def ensure_cluster_weights(df: pd.DataFrame) -> None:
    """Create / refresh cluster_stats + kernel_bandwidth inside WEIGHTS_DIR."""
    stats_p = WEIGHTS_DIR / "cluster_stats.npz"
    kern_p = WEIGHTS_DIR / "kernel_bandwidth.json"

    # Re-use if already up-to-date -----------------------------------
    if stats_p.exists() and kern_p.exists():
        try:
            with np.load(stats_p) as st:
                _ = st["mu"], st["var"]
            return
        except KeyError:
            log.warning("Out-of-date cluster_stats.npz detected – rebuilding.")

    # Build sample ----------------------------------------------------
    num = (
        df.select_dtypes("number")
        .drop(columns=["open", "high", "low", "close", "volume"], errors="ignore")
    )
    sample = num.sample(min(25_000, len(num)), random_state=42)

    # Split into X (features) and y (outcome) -------------------------
    if "future_return" in sample.columns:
        y = sample.pop("future_return").to_numpy(np.float32)
    else:
        first = sample.columns[0]
        y = sample.pop(first).to_numpy(np.float32)          # pseudo-outcome
    X = sample.to_numpy(np.float32)
    feature_names = list(sample.columns)

    k = max(3, min(15, len(sample) // 2))
    PathClusterEngine.build(
        X,
        y,
        feature_names=feature_names,
        n_clusters=k,
        out_dir=WEIGHTS_DIR,
        random_state=42,
    )
    log.info("✓ Rebuilt cluster_stats.npz (k=%d)", k)



# ---------------------------------------------------------------------------
# 3) Optional RF feature‑weights
# ---------------------------------------------------------------------------

def find_rf_weights() -> Path | None:
    rf = next(WEIGHTS_DIR.glob("rf_feature_weights*.pkl"), None)
    if not rf:
        log.warning("! RF feature weights not found – EVEngine will run KNN‑only")
    return rf


# ---------------------------------------------------------------------------
# 4) Main driver
# ---------------------------------------------------------------------------

def main() -> None:
    pq_path = find_parquet()
    log.info("Parquet slice → %s", pq_path.relative_to(ROOT))

    df = pd.read_parquet(pq_path)


    # ------------------------------------------------------------------
    # Latency monitor stub + safety-config for ExecutionManager
    # ------------------------------------------------------------------
    lat_monitor = SimpleNamespace(mean=0.0)   # replace with real monitor later

    config = {                               # five-tier Safety FSM defaults
        "safety": {
            "latency_ms": 300,
            "single_loss_pct": 1.0,
            "daily_loss_pct": 3.0,
            "drawdown_pct": 6.0,
            "vix_spike": 0.10,
        }
    }

    # ------------------------------------------------------------------
    # Ensure a 'symbol' column exists – infer from partition if absent
    # ------------------------------------------------------------------
    if "symbol" not in df.columns:
        sym_val = next(
            (part.split("=")[1] for part in pq_path.parts if part.startswith("symbol=")),
            "UNK",
        )
        df["symbol"] = sym_val
        log.warning("Added missing 'symbol' column with value %s", sym_val)

    #stats_p, kern_p = ensure_cluster_weights(df)
    #rf_p = find_rf_weights()

    #ev = EVEngine.from_artifacts(stats_p, kern_p, rf_p, knn_only=(rf_p is None))

    # ---- build / load artefacts directory ----
        ensure_cluster_weights(df)  # writes into WEIGHTS_DIR
        metric = "rf_weighted" if find_rf_weights() else "euclidean"

        ev = EVEngine.from_artifacts(
                WEIGHTS_DIR,
                metric = metric,
        )

    risk = RiskManager(100_000, max_leverage=25_000)
    #exec_mgr = ExecutionManager(ev, risk, ROOT / "logs" / "signals.jsonl")
    exec_mgr = ExecutionManager(
        ev,                     # EVEngine instance
        risk,                   # RiskManager instance
        lat_monitor=lat_monitor,
        config=config,          # safety config
        log_path=ROOT / "logs" / "signals.jsonl",
    )

    broker = BrokerStub(slippage_bp=2.0)

    bt = AsyncBacktester(exec_mgr, df, broker, equity0=100_000)
    summary = asyncio.run(bt.run())

    log.info("Summary → %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        log.exception("Fatal error", exc_info=exc)
        sys.exit(1)