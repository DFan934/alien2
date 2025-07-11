# ---------------------------------------------------------------------------
# FILE: prediction_engine/scripts/run_backtest.py
# ---------------------------------------------------------------------------
"""
Offline batch back-test of EVEngine on historical minute bars.

Outputs
-------
• CSV with per-bar signals & P&L
• Console summary of total P&L
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering.pipelines.core import CoreFeaturePipeline
from prediction_engine.ev_engine import EVEngine
from prediction_engine.execution.risk_manager import RiskManager

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG – edit here and just press ▶︎
# ──────────────────────────────────────────────────────────────────────────────
CONFIG: dict[str, object] = {
    "csv": "raw_data/RRC.csv",              # relative OR absolute
    "symbol": "RRC",
    "start": "1998-08-26",
    "end":   "1998-11-26",
    "equity": 100_000.0,
    "out": "backtest_signals.csv",
}
# ──────────────────────────────────────────────────────────────────────────────


def _resolve_csv(path_like: str | Path) -> Path:
    """
    Resolve *csv_path* robustly:

    1. Treat it as absolute / user-expanded first.
    2. If not found, assume it is relative to the **project root**
       (two levels above this script).
    """
    p = Path(path_like).expanduser().resolve()
    if p.is_file():
        return p

    project_root = Path(__file__).resolve().parents[2]  # “…/prediction_engine/…”
    alt = (project_root / path_like).resolve()
    if alt.is_file():
        return alt

    raise FileNotFoundError(f"CSV not found at '{p}' or '{alt}'")


def run(cfg: dict[str, object]) -> None:
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    # 1) Load & clean CSV ─────────────────────────────────────────────
    csv_path = _resolve_csv(cfg["csv"])
    df_raw = pd.read_csv(csv_path)

    # merge Date + Time → timestamp (avoids parse_dates FutureWarning)
    df_raw["timestamp"] = pd.to_datetime(df_raw["Date"] + " " + df_raw["Time"])
    df_raw.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        },
        inplace=True,
    )
    df_raw["symbol"] = cfg["symbol"]
    df_raw = df_raw[
        (df_raw["timestamp"] >= cfg["start"]) & (df_raw["timestamp"] <= cfg["end"])
    ].sort_values("timestamp", ignore_index=True)

    # 2) Feature engineering ─────────────────────────────────────────
    pipe = CoreFeaturePipeline(parquet_root=Path("."))  # dummy root – in-mem DF
    feats, _ = pipe.run_mem(df_raw)
    print("Feature shape:", feats.shape)

    # 3) Dummy EVEngine (replace with real artefacts if you have them)
    feature_cols = [c for c in feats.columns if c not in ("symbol", "timestamp")]

    feature_dim = feats.shape[1] - 2  # subtract 'symbol', 'timestamp' cols
    centers = np.zeros((2, feature_dim), dtype=np.float32)
    mu = np.zeros(2, dtype=np.float32)
    var = np.ones(2, dtype=np.float32) * 0.001
    ev = EVEngine(centers=centers, mu=mu, var=var, var_down=var, h=1.0)


    risk = RiskManager(account_equity=float(cfg["equity"]))

    # 4) Iterate bars & collect results ──────────────────────────────
    numeric_cols = feats.select_dtypes("number").columns.tolist()


    out_rows: list[dict] = []
    for _, row in feats.iterrows():
        x_num = row[numeric_cols].to_numpy(np.float32)        # ← fixed line
        ev_res = ev.evaluate(x_num, adv_percentile=10.0)
        qty = risk.desired_size(cfg["symbol"], row["close"])
        pnl = qty * ev_res.mu
        risk.on_closed_trade(pnl)

        out_rows.append(
            {
                "ts": row["timestamp"],
                "µ": ev_res.mu,
                "σ²": ev_res.sigma,
                "kelly_qty": qty,
                "pnl": pnl,
                "cum_pnl": risk.account_equity - cfg["equity"],
            }
        )

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(cfg["out"], index=False)
    log.info("Saved signals → %s", cfg["out"])
    log.info("Total P&L: $%.2f", out_df['cum_pnl'].iloc[-1])


if __name__ == "__main__":
    run(CONFIG)
