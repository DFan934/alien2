# ---------------------------------------------------------------------------
# FILE: prediction_engine/scripts/run_backtest.py
# ---------------------------------------------------------------------------
"""
Event-driven batch back-test of EVEngine on historical OHLCV bars.

Outputs
-------
• CSV (signals_out) with per-bar signals, fills & PnL
• Console summary of cumulative P&L + risk metrics
"""
from __future__ import annotations
import pandas as pd
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

#import pandas as pd

from feature_engineering.pipelines.core import CoreFeaturePipeline
from prediction_engine.ev_engine import EVEngine
from prediction_engine.tx_cost import BasicCostModel  # NEW
from execution.risk_manager import RiskManager
from prediction_engine.testing_validation.backtester import BrokerStub  # NEW
from scripts.rebuild_artefacts import rebuild_if_needed  # NEW
from backtester import Backtester
from execution.manager import ExecutionManager
from execution.metrics.report import load_blotter, pnl_curve, latency_summary


# Keep only PCA feature columns (produced by CoreFeaturePipeline)
def _pca_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("pca_")]


# ────────────────────────────────────────────────────────────────────────
# CONFIG – edit here
# ────────────────────────────────────────────────────────────────────────
CONFIG: Dict[str, Any] = {
    # raw minute-bar CSV (Date, Time, Open, High, Low, Close, Volume)
    "csv": "raw_data/RRC.csv",
    "parquet_root": "parquet",
    "symbol": "RRC",
    "start": "1998-08-26",
    "end": "1998-11-26",

    # artefacts created by PathClusterEngine.build()
    "artefacts": "../weights",

    # capital & trading costs
    "equity": 100_000.0,
    # "spread_bp": 2.0,          # half-spread in basis points
    # "commission": 0.002,       # $/share
    # "slippage_bp": 0.0,        # BrokerStub additional bp slippage

    "spread_bp": 0.0,  # half-spread in basis points
    "commission": 0.0,  # $/share
    "slippage_bp": 0.0,  # BrokerStub additional bp slippage

    # misc
    "out": "backtest_signals.csv",
    "atr_period": 14,
}


# ────────────────────────────────────────────────────────────────────────


def _resolve_path(path_like: str | Path) -> Path:
    """Resolve path relative to project root if not absolute."""
    # 1) absolute or cwd-relative
    p = Path(path_like).expanduser()
    if p.exists():
        return p.resolve()
    # 2) try relative to the repository root (two levels up)
    repo_root = Path(__file__).parents[1]
    alt = (repo_root / path_like).expanduser()
    if alt.exists():
        return alt.resolve()
    raise FileNotFoundError(f"Could not find {path_like!r} in CWD or under {repo_root!s}")


# ────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────
def run(cfg: Dict[str, Any]) -> pd.DataFrame:
    logging.basicConfig(
        level=logging.INFO, format="[%(@levelname)s] %(message)s", stream=sys.stdout
    )
    # suppress EVEngine’s regime/residual warnings
    # logging.getLogger("prediction_engine.ev_engine").setLevel(logging.ERROR)

    log = logging.getLogger("backtest")
    import pandas as pd
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

    # ------------------------------------------------------------------
    #  Solution: Add the missing 'trigger_ts' column here
    # ------------------------------------------------------------------
    df_raw["trigger_ts"] = df_raw["timestamp"]

    # ------------------------------------------------------------------
    #  Solution: Calculate and add the 'volume_spike_pct' column
    # ------------------------------------------------------------------
    # Calculate a 20-period moving average of volume
    volume_ma = df_raw['volume'].rolling(window=20, min_periods=1).mean()
    # Calculate the spike as a percentage change from the moving average
    df_raw['volume_spike_pct'] = (df_raw['volume'] / volume_ma) - 1.0
    # Fill any initial NaN values (from the rolling window) with 0.0
    #df_raw['volume_spike_pct'].fillna(0.0, inplace=True)
    # Fill any initial NaN values (from the rolling window) with 0.0
    df_raw['volume_spike_pct'] = df_raw['volume_spike_pct'].fillna(0.0)

    # 2 ─ Feature engineering
    pipe = CoreFeaturePipeline(parquet_root=Path(""))  # in-mem
    feats, _ = pipe.run_mem(df_raw)



    # ------------------------------------------------------------------
    # keep only PCA columns (+ ids)  ────────────
    # ------------------------------------------------------------------
    pc_cols = _pca_cols(feats)
    if not pc_cols:
        raise RuntimeError("No pca_* columns found – did CoreFeaturePipeline run?")
    feats = feats[pc_cols + ["symbol", "timestamp"]]
    feats["open"] = df_raw["open"].values  # Attach open for execution logic
    feats["close"] = df_raw["close"].values

    # sanity: ensure ATR exists
    ATR_COL = f"atr{cfg['atr_period']}"
    if ATR_COL not in feats.columns:
        feats[ATR_COL] = (
                df_raw["high"]
                .rolling(cfg['atr_period'])
                .max()
                - df_raw["low"].rolling(cfg['atr_period']).min()
        ).bfill()

    # 3 ─ Load EVEngine artefacts
    art_dir = _resolve_path(cfg["artefacts"])

    # resolve parquet directory (same repo root logic)
    pq_root = _resolve_path(cfg.get("parquet_root", "parquet"))

    # if it contains a "symbol=…" subfolder, use that (so we don't include schema.json)
    sym_dir = pq_root / f"symbol={cfg['symbol']}"
    if sym_dir.exists() and sym_dir.is_dir():
        pq_root = sym_dir

    # ── convert start/end to real Timestamps for Parquet filtering ─────
    from pandas import to_datetime
    start_ts = to_datetime(cfg["start"])
    end_ts = to_datetime(cfg["end"])

    rebuild_if_needed(
        artefact_dir=cfg["artefacts"],
        parquet_root=str(pq_root),  # ← pass the resolved path, not the literal key
        symbols=[cfg["symbol"]],
        start=start_ts,
        end=end_ts,
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

    # feature_cols = [
    #    c for c in feats.columns if c not in ("symbol", "timestamp")
    # ]  # order preserved
    # numeric_idx = feats[feature_cols].select_dtypes("number").columns.tolist()

    # 4 ─ Instantiate Broker, Cost model & RiskManager

    cost_model = BasicCostModel()
    cost_model.spread_bp = float(cfg["spread_bp"])
    cost_model.commission = float(cfg["commission"])
    broker = BrokerStub(slippage_bp=float(cfg["slippage_bp"]))
    # risk = RiskManager(
    #    account_equity=float(cfg["equity"]), cost_model=cost_model, max_kelly=0.5
    # )

    # RiskManager now only takes initial equity and max_kelly; drop cost_model
    risk = RiskManager(
        float(cfg["equity"]))

    # cost_model = BasicCostModel(
    #    spread_bp=float(cfg["spread_bp"]), commission=float(cfg["commission"])
    # )
    # broker = BrokerStub(slippage_bp=float(cfg["slippage_bp"]))
    # risk = RiskManager(
    #    account_equity=float(cfg["equity"]), cost_model=cost_model, max_kelly=0.5
    # )

    # 5 ─ Event loop
    bt = Backtester(feats, cfg)  # Pass the prepared DataFrame

    eq_curve = bt.run()

    csv_path = _resolve_path(cfg["csv"])
    df_raw = pd.read_csv(csv_path)
    # ─── rename uppercase OHLC to lowercase to match resample keys ───
    df_raw.rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close"},
        inplace=True,
    )

    # ─── Recompute daily regimes for later breakdown ───
    from prediction_engine.market_regime import label_days, RegimeParams
    df_raw["timestamp"] = pd.to_datetime(df_raw["Date"] + " " + df_raw["Time"])
    daily_df = (
        df_raw
        .set_index("timestamp")
        .resample("D")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    daily_regimes = label_days(daily_df, RegimeParams())

    # Save
    out_path = Path(cfg["out"])
    eq_curve.to_csv(out_path, index=False)

    import matplotlib.pyplot as plt

    # After eq_curve.to_csv(...)
    df = pd.read_csv(out_path)

    # equity‐based cumulative return
    df["cum_return"] = df["equity"] / cfg["equity"] - 1.0

    # --- A) Add one‑bar forward return and win/loss stats ---
    signals = df.copy()
    signals["ret1m"] = signals["open"].shift(-1) / signals["open"] - 1.0

    # mask for all trades you took:
    trade_mask = signals.qty_next_bar > 0
    # among those, which were wins?
    winning_mask = (signals.ret1m > 0)

    print("Positive‑µ trade stats:")
    print("  Total trades:", trade_mask.sum())
    print("  Win rate:   ", (trade_mask & winning_mask).sum() / trade_mask.sum())
    # average return *of the trades you actually took*:
    avg_ret = signals.loc[trade_mask, "ret1m"].mean()
    print("  Avg ret:    ", avg_ret)


    from scripts.diagnostics import BacktestDiagnostics

    diag = BacktestDiagnostics(
        df=eq_curve,
        raw=df_raw,
        cfg=cfg
    )
    diag.run_all()

    plt.figure()
    plt.scatter(signals.mu, signals.ret1m, s=5, alpha=0.3)
    plt.title("µ forecast vs realized 1‑bar return")
    plt.xlabel("µ")
    plt.ylabel("ret1m")
    plt.grid(True)
    plt.show()

    # --- C) Cluster‑level summary ---
    cluster_stats = df.groupby("cluster_id").apply(
        lambda g: pd.Series({
            "n": len(g),
            "mean_µ": g["mu"].mean(),
            "mean_ret1m": (g["open"].shift(-1) / g["open"] - 1.0).mean()
        })
    ).sort_values("mean_ret1m")
    print(cluster_stats.head(10))

    # buy‐and‐hold using raw data (df_raw) rather than the signals CSV
    buy_hold = df_raw["close"].iloc[-1] / df_raw["open"].iloc[0] - 1.0
    print(f"Buy‑and‑Hold over period: {buy_hold * 100:.2f}%")
    df["regime"] = pd.to_datetime(df["timestamp"]).dt.normalize().map(daily_regimes)
    print(df.groupby("regime")["cum_return"].agg(["min", "max", "mean"]))

    plt.figure()
    plt.plot(df["timestamp"], df["cum_return"], lw=1)
    plt.title("Cumulative Return Over Time")
    plt.xlabel("Time")
    plt.ylabel("Return")
    plt.grid(True)
    plt.show()

    # 1. Print statistical summary
    print("\nDiagnostic summary:")
    print(df[["mu", "residual", "position_size"]].describe())

    # ** Print a nice summary before returning **
    final_eq = float(eq_curve["equity"].iloc[-1])
    total_ret = 100 * (final_eq / cfg["equity"] - 1)
    log.info("Backtest complete.  Equity curve → %s", out_path)
    log.info("  Final equity: $%.2f   Total return: %.2f%%   Bars: %d",
             final_eq, total_ret, len(eq_curve))

    # 2. Plot histograms (one per figure)
    for col in ("mu", "residual", "position_size"):
        plt.figure()
        df[col].hist(bins=50)
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.show()

    return eq_curve


__all__ = ["run", "CONFIG"]

if __name__ == "__main__":
    run(CONFIG)




