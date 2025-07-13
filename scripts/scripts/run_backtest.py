# ---------------------------------------------------------------------------
# FILE: prediction_engine/scripts/run_backtest.py
# ---------------------------------------------------------------------------
"""
Event-driven batch back-test of EVEngine on historical OHLCV bars.

Outputs
-------
• CSV (`signals_out`) with per-bar signals, fills & PnL
• Console summary of cumulative P&L + risk metrics
"""
from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from feature_engineering.pipelines.core import CoreFeaturePipeline
from prediction_engine.ev_engine import EVEngine
from prediction_engine.tx_cost import BasicCostModel        # NEW
from prediction_engine.execution.risk_manager import RiskManager
from prediction_engine.testing_validation.backtester import BrokerStub #NEW
from prediction_engine.scripts.rebuild_artefacts import rebuild_if_needed   #  NEW
from scanner.schema import FEATURE_ORDER


# Keep only PCA feature columns (produced by CoreFeaturePipeline)
def _pca_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("pca_")]


# ────────────────────────────────────────────────────────────────────────
# CONFIG – edit here
# ────────────────────────────────────────────────────────────────────────
CONFIG: Dict[str, Any] = {
    # raw minute-bar CSV (Date, Time, Open, High, Low, Close, Volume)
    "csv": "raw_data/RRC.csv",
    "symbol": "RRC",
    "start": "1998-08-26",
    "end":   "1998-11-26",

    # artefacts created by PathClusterEngine.build()
    "artefacts": "weights",

    # capital & trading costs
    "equity": 100_000.0,
    "spread_bp": 2.0,          # half-spread in basis points
    "commission": 0.002,       # $/share
    "slippage_bp": 0.0,        # BrokerStub additional bp slippage

    # misc
    "out": "backtest_signals.csv",
    "atr_period": 14,
}
# ────────────────────────────────────────────────────────────────────────


def _resolve_path(path_like: str | Path) -> Path:
    """Resolve path relative to project root if not absolute."""
    p = Path(path_like).expanduser().resolve()
    if p.exists():
        return p
    root = Path(__file__).resolve().parents[2]
    q = (root / path_like).resolve()
    if q.exists():
        return q
    raise FileNotFoundError(path_like)


# ────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────
def run(cfg: Dict[str, Any]) -> None:
    logging.basicConfig(
        level=logging.INFO, format="[%(@levelname)s] %(message)s", stream=sys.stdout
    )
    log = logging.getLogger("backtest")

    # 1 ─ Load raw CSV
    csv_path = _resolve_path(cfg["csv"])
    df_raw = pd.read_csv(csv_path)

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
    df_raw = (
        df_raw[
            (df_raw["timestamp"] >= cfg["start"]) & (df_raw["timestamp"] <= cfg["end"])
        ]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    if df_raw.empty:
        raise ValueError("Date filter returned zero rows")

    # 2 ─ Feature engineering
    pipe = CoreFeaturePipeline(parquet_root=Path("."))  # in-mem
    feats, _ = pipe.run_mem(df_raw)

    # ------------------------------------------------------------------
    # keep only PCA columns (+ ids)  ────────────
    # ------------------------------------------------------------------
    pc_cols = _pca_cols(feats)
    if not pc_cols:
        raise RuntimeError("No pca_* columns found – did CoreFeaturePipeline run?")
    feats = feats[pc_cols + ["symbol", "timestamp"]]
    feats["open"] = df_raw["open"].values  # Attach open for execution logic

    # sanity: ensure ATR exists
    ATR_COL = f"atr{cfg['atr_period']}"
    if ATR_COL not in feats.columns:
        feats[ATR_COL] = (
            df_raw["high"]
            .rolling(cfg["atr_period"])
            .max()
            - df_raw["low"].rolling(cfg["atr_period"]).min()
        ).fillna(method="bfill")

    # 3 ─ Load EVEngine artefacts
    art_dir = _resolve_path(cfg["artefacts"])

    rebuild_if_needed(
        artefact_dir=cfg["artefacts"],
        parquet_root="parquet/minute_bars",  # <-- adjust to your dataset root
        symbols=[cfg["symbol"]],
        start=cfg["start"],
        end=cfg["end"],
        n_clusters=64,
    )


    ev = EVEngine.from_artifacts(art_dir)

    # schema guard – compare EXACT list of pca_* columns in order
    with open(art_dir / "meta.json", "r", encoding="utf-8") as mf:
        meta = json.load(mf)
    expected = meta["features"]  # e.g. ["pca_1","pca_2",…]

    print("\n[DEBUG] Features listed in meta.json (artefacts):", expected)
    print("[DEBUG] Features produced at runtime (pc_cols):", pc_cols)
    print("[DEBUG] Are they equal?", expected == pc_cols)

    if expected != pc_cols:
        raise RuntimeError(
            f"\n[ERROR] Feature list mismatch.\n  Artefacts: {expected}\n  Runtime : {pc_cols}"
        )

    numeric_idx = pc_cols  # only PCA features go to EVEngine

    '''sha_actual = hashlib.sha1(sha_actual.encode()).hexdigest()[:12]
    if sha_actual != sha_expected:
        print("\n[DEBUG] Features in artefacts (meta.json):")
        with open(art_dir / "meta.json", "r", encoding="utf-8") as mf:
            meta = json.load(mf)
            print(meta["features"])

        print("\n[DEBUG] Features produced at runtime (feats):")
        feature_cols_live = [c for c in feats.columns if c not in ("symbol", "timestamp")]
        print(feature_cols_live)

        raise RuntimeError(
            f"Feature SHA mismatch – artefact built for {sha_expected}, got {sha_actual}"
        )'''

    #feature_cols = [
    #    c for c in feats.columns if c not in ("symbol", "timestamp")
    #]  # order preserved
    #numeric_idx = feats[feature_cols].select_dtypes("number").columns.tolist()

    # 4 ─ Instantiate Broker, Cost model & RiskManager

    cost_model = BasicCostModel()
    cost_model.spread_bp = float(cfg["spread_bp"])
    cost_model.commission = float(cfg["commission"])
    broker = BrokerStub(slippage_bp=float(cfg["slippage_bp"]))
    #risk = RiskManager(
    #    account_equity=float(cfg["equity"]), cost_model=cost_model, max_kelly=0.5
    #)

    # RiskManager now only takes initial equity and max_kelly; drop cost_model
    risk = RiskManager(
        float(cfg["equity"]))

    #cost_model = BasicCostModel(
    #    spread_bp=float(cfg["spread_bp"]), commission=float(cfg["commission"])
    #)
    #broker = BrokerStub(slippage_bp=float(cfg["slippage_bp"]))
    #risk = RiskManager(
    #    account_equity=float(cfg["equity"]), cost_model=cost_model, max_kelly=0.5
    #)

    # 5 ─ Event loop
    rows_out: List[Dict[str, Any]] = []

    for i, row in feats.iterrows():
        bar = {
            "ts": row["timestamp"],
            "symbol": row["symbol"],
            "price": row["open"],          # fills at NEXT bar's open
        }

        # ─ 5.1 execute any queued orders (from previous bar)
        fills = broker.execute_pending(bar)
        realised_pnl = 0.0

        for tr in fills:
            # tr is a dict (or object) containing your fill info.
            # extract the float PnL before handing it to RiskManager:

            if isinstance(tr, dict):
                pnl = tr.get("pnl", tr.get("realised_pnl", 0.0))
            else:
                pnl = getattr(tr, "pnl", 0.0)

            # update RiskManager’s equity and collect for logging
            risk.on_closed_trade(pnl)
            realised_pnl += pnl
        # ─ 5.2 unrealised P/L
        mtm = broker.mark_to_market(bar)

        # ─ 5.3 generate signal for NEXT bar
        x = row[pc_cols].to_numpy(np.float32)
        ev_res = ev.evaluate(x, adv_percentile=10.0)

        atr_val = row[ATR_COL]
        #qty = risk.desired_size(cfg["symbol"], price=row["open"], atr=atr_val)

        # desired_size no longer takes keywords – pass positionally:
        qty = risk.desired_size(cfg["symbol"], row["open"])

        if qty != 0:
            broker.queue_order(cfg["symbol"], qty)

        rows_out.append(
            {
                "ts": bar["ts"],
                "µ": ev_res.mu,
                "σ²": ev_res.sigma,
                "qty_next_bar": qty,
                "realised_pnl": realised_pnl,
                "unrealised_pnl": mtm,
                "equity": risk.account_equity + mtm,
            }
        )

    out_df = pd.DataFrame(rows_out)
    out_path = Path(cfg["out"])
    out_df.to_csv(out_path, index=False)
    log.info("Saved bar-level signals → %s", out_path)

    log.info(
        "Final equity: $%.2f  |  Total return: %.2f %%  |  Bars: %d",
        out_df["equity"].iloc[-1],
        (out_df["equity"].iloc[-1] / cfg["equity"] - 1) * 100,
        len(out_df),
    )


if __name__ == "__main__":
    run(CONFIG)
