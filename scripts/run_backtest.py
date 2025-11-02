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
from data_ingestion.manifest import summarize_partitions_fast

import asyncio
# after: from prediction_engine.calibration import load_calibrator, calibrate_isotonic
#from ledger import PortfolioLedger, equity_curve_from_trades
import glob

import pyarrow.dataset as ds
from prediction_engine.prediction_engine.artifacts.manager import ArtifactManager

import numpy as np
import pandas as pd
import json
from scripts.a2_report import generate_report  # NEW
from sklearn.isotonic import IsotonicRegression

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List
from prediction_engine.testing_validation.walkforward import WalkForwardRunner

from prediction_engine.calibration import load_calibrator, map_mu_to_prob
# Optional: only needed when building a portfolio equity curve in Phase-4.
# Keep import lazy-safe so tests can import this module without the ledger package.
try:
    from ledger import PortfolioLedger, equity_curve_from_trades  # type: ignore
except Exception:  # ModuleNotFoundError or anything else
    PortfolioLedger = None  # not used in this file; stub to avoid NameError
    def equity_curve_from_trades(trades_df):
        """
        Minimal fallback: build an equity curve from realized PnL if present.
        Returns a Pandas Series indexed by exit_ts with cumulative equity (starting at 0).
        Used only in Phase-4; tests that import this module won't touch it.
        """
        import pandas as _pd
        if not isinstance(trades_df, _pd.DataFrame):
            return _pd.Series(dtype=float)
        if "exit_ts" not in trades_df.columns or "realized_pnl" not in trades_df.columns:
            return _pd.Series(dtype=float)
        df = trades_df[["exit_ts", "realized_pnl"]].copy()
        df["exit_ts"] = _pd.to_datetime(df["exit_ts"], utc=True, errors="coerce")
        df = df.dropna(subset=["exit_ts", "realized_pnl"]).sort_values("exit_ts")
        return df.set_index("exit_ts")["realized_pnl"].cumsum().rename("equity")
from feature_engineering.pipelines.core import CoreFeaturePipeline
from prediction_engine.ev_engine import EVEngine
from prediction_engine.tx_cost import BasicCostModel  # NEW
from execution.risk_manager import RiskManager
from prediction_engine.testing_validation.async_backtester import BrokerStub  # NEW
from scripts.rebuild_artefacts import rebuild_if_needed  # NEW
from scanner.detectors import build_detectors        # ‹— add scanner import
from backtester import Backtester
from execution.manager import ExecutionManager
from execution.metrics.report import load_blotter, pnl_curve, latency_summary
from execution.manager import ExecutionManager
from execution.risk_manager import RiskManager            # NEW
from prediction_engine.tx_cost import BasicCostModel
from prediction_engine.calibration import load_calibrator, calibrate_isotonic  # NEW – iso µ‑cal


# ─── global run metadata ─────────────────────────────────────────────
import uuid, subprocess, datetime as dt






#from universes import StaticUniverse, FileUniverse
#from universes import resolve_universe, UniverseError

from universes.providers import StaticUniverse, FileUniverse
import universes.providers as U


import hashlib, json

def _stable_universe_hash(symbols: list[str]) -> str:
    """Stable SHA1 over sorted symbols; safe for logs & meta."""
    payload = json.dumps(sorted([str(s).upper() for s in symbols]), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

def _universe_source_label(cfg: dict) -> str:
    u = cfg.get("universe")
    try:
        # StaticUniverse / FileUniverse instances (providers module)
        if isinstance(u, U.StaticUniverse):
            return "static:list"
        if isinstance(u, U.FileUniverse):
            return f"file:{getattr(u, 'path', '(unknown)')}"
        # raw shapes users might pass
        if isinstance(u, (list, tuple)):
            return "static:list"
        if isinstance(u, str):
            return f"file:{u}"
        if isinstance(u, dict) and u.get("type", "").lower() == "sp500":
            return f"sp500:{u.get('as_of','')}"
    except Exception:
        pass
    return "unknown"



RUN_ID = uuid.uuid4().hex[:8]          # short random id
try:
    '''GIT_SHA = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=Path(__file__).parents[2],  # repo root
    ).decode().strip()'''
    GIT_SHA = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=Path(__file__).parents[2],
        stderr=subprocess.DEVNULL,
    ).decode().strip()

except Exception:
    GIT_SHA = "n/a"
RUN_META = {
    "run_id": RUN_ID,
    "git_sha": GIT_SHA,
    "started": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
}

# AFTER: RUN_META = {...}
# ---- Step-3: mode toggle for backtest runner ----
BACKTEST_MODE: str = "CLI"  # or "NOCLI" to auto-run all symbols across entire time range

from feature_engineering.pipelines.dataset_loader import load_parquet_dataset  # reuse loader types
import pyarrow.dataset as ds


# ─────────────────────────────────────────────────────────────────────



# Keep only PCA feature columns (produced by CoreFeaturePipeline)
def _pca_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("pca_")]



from pathlib import Path

def discover_parquet_symbols(parquet_root: Path) -> list[str]:
    syms = []
    for p in parquet_root.glob("symbol=*"):
        if p.is_dir():
            syms.append(p.name.split("=", 1)[-1].upper())
    return sorted(set(syms))


# AFTER: def discover_parquet_symbols(parquet_root: Path) -> list[str]:
def _discover_time_bounds_all(input_root: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    dataset = ds.dataset(str(input_root), format="parquet", partitioning="hive", exclude_invalid_files=True)
    scanner = dataset.scanner(columns=["timestamp"])
    tbl = scanner.to_table()
    s = pd.to_datetime(tbl.column("timestamp").to_pandas(), utc=True)
    return s.min(), s.max()


# AFTER: def _discover_time_bounds_all(input_root: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
from dataclasses import dataclass
from typing import List, Tuple, Literal

@dataclass(frozen=True)
class ResolvedUniverseWindow:
    symbols: List[str]
    start: str
    end: str
    partitions: str  # e.g., "BBY:2, RRC:1"

def _partition_counts_str(input_root: Path, symbols: List[str]) -> str:
    parts = []
    for s in symbols:
        p = input_root / f"symbol={s}"
        n = len(list(p.glob("year=*/month=*")))
        parts.append((s, n))
    parts.sort(key=lambda x: x[0])
    return ", ".join([f"{s}:{n}" for s, n in parts])

def resolve_universe_window_for_tests(
    parquet_root: str | Path,
    cfg: dict,
    mode: Literal["CLI","NOCLI"] = "CLI",
) -> ResolvedUniverseWindow:
    """
    Pure, side-effect-free resolver used by tests to check Step-3 wiring.
    Decides symbols/start/end the same way the runner does, and returns a partition summary.
    Does NOT kick off a backtest.
    """
    root = Path(parquet_root)
    if mode.upper() == "NOCLI":
        syms = discover_parquet_symbols(root)
        start_ts, end_ts = _discover_time_bounds_all(root)
        start_str = start_ts.strftime("%Y-%m-%d")
        end_str   = end_ts.strftime("%Y-%m-%d")
    else:
        # CLI mode: take what's in cfg if provided, else discover symbols; keep dates if provided
        syms = cfg.get("universe").symbols if hasattr(cfg.get("universe"), "symbols") else cfg.get("symbols")
        if not syms:
            syms = discover_parquet_symbols(root)
        start_str = cfg.get("start") or "1900-01-01"
        end_str   = cfg.get("end")   or "2100-01-01"

    return ResolvedUniverseWindow(
        symbols=list(syms),
        start=start_str,
        end=end_str,
        partitions=_partition_counts_str(root, syms),
    )



from typing import Tuple
import glob

def _consolidate_phase4_outputs(artifacts_root: Path) -> Tuple[Path | None, Path | None]:
    """
    Gather per-fold/per-symbol outputs and write:
      artifacts_root/decisions.parquet
      artifacts_root/trades.parquet
    Returns (decisions_path, trades_path) — either may be None if nothing found.
    """
    artifacts_root = artifacts_root.resolve()
    decisions_out = artifacts_root / "decisions.parquet"
    trades_out    = artifacts_root / "trades.parquet"

    # --- decisions: accept parquet or csv from folds ---
    dec_parts = []
    # common names we’ll try (be lenient with globbing)
    dec_patterns = [
        str(artifacts_root / "**" / "decisions.parquet"),
        str(artifacts_root / "**" / "decisions_*.parquet"),
        str(artifacts_root / "**" / "decisions.csv"),
        str(artifacts_root / "**" / "decisions_*.csv"),
        str(artifacts_root / "**" / "signals.parquet"),  # some repos use 'signals'
        str(artifacts_root / "**" / "signals_*.parquet"),
        str(artifacts_root / "**" / "signals.csv"),
        str(artifacts_root / "**" / "signals_*.csv"),
    ]
    for pat in dec_patterns:
        for fp in glob.glob(pat, recursive=True):
            try:
                if fp.endswith(".parquet"):
                    dec_parts.append(pd.read_parquet(fp))
                elif fp.endswith(".csv"):
                    dec_parts.append(pd.read_csv(fp))
            except Exception:
                pass

    if dec_parts:
        dec = pd.concat(dec_parts, ignore_index=True).drop_duplicates()
        # ensure required cols are present if we can
        # (if some folds miss columns, outer-join behavior of concat leaves NaNs — acceptable)
        dec.to_parquet(decisions_out, index=False)
        decisions_path = decisions_out
    else:
        decisions_path = None

    # --- trades/fills/blotter: pick whatever exists and unify ---
    trade_parts = []
    trade_patterns = [
        str(artifacts_root / "**" / "trades.parquet"),
        str(artifacts_root / "**" / "trades_*.parquet"),
        str(artifacts_root / "**" / "fills.parquet"),
        str(artifacts_root / "**" / "fills_*.parquet"),
        str(artifacts_root / "**" / "blotter.parquet"),
        str(artifacts_root / "**" / "blotter_*.parquet"),
        str(artifacts_root / "**" / "trades.csv"),
        str(artifacts_root / "**" / "fills.csv"),
        str(artifacts_root / "**" / "blotter.csv"),
    ]
    for pat in trade_patterns:
        for fp in glob.glob(pat, recursive=True):
            try:
                if fp.endswith(".parquet"):
                    trade_parts.append(pd.read_parquet(fp))
                elif fp.endswith(".csv"):
                    trade_parts.append(pd.read_csv(fp))
            except Exception:
                pass

    if trade_parts:
        trades = pd.concat(trade_parts, ignore_index=True).drop_duplicates()
        trades.to_parquet(trades_out, index=False)
        trades_path = trades_out
    else:
        trades_path = None

    return decisions_path, trades_path


'''def _resolve_universe(obj) -> List[str]:
    # Accept StaticUniverse, FileUniverse, or a plain list of strings
    if isinstance(obj, StaticUniverse):
        return obj.list()
    if isinstance(obj, FileUniverse):
        return obj.list()
    if isinstance(obj, (list, tuple)):
        return sorted({str(s).strip().upper() for s in obj})
    raise TypeError(f"Unsupported universe type: {type(obj)}")
'''






def _load_bars_for_symbol(cfg: Dict[str, Any], symbol: str) -> pd.DataFrame:
    """
    Load OHLCV bars for one symbol and [start,end].
    Prefers CSV if cfg["csv"] points to an existing file FOR THIS SYMBOL.
    Otherwise, read from hive-partitioned parquet root.
    """
    start, end = cfg["start"], cfg["end"]
    # Try CSV first if user points to a per-symbol file like raw_data/RRC.csv
    csv_path = cfg.get("csv")
    if csv_path:
        p = _resolve_path(csv_path)
        if p.exists():
            df = pd.read_csv(p)
            # Expect columns Date, Time, Open, High, Low, Close, Volume
            ts = pd.to_datetime(df["Date"] + " " + df["Time"])
            df = df.rename(columns={
                "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
            })
            df["timestamp"] = ts
            df["symbol"] = symbol
            df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)].sort_values("timestamp").reset_index(drop=True)
            if not df.empty:
                return df

    # Fallback: hive-partitioned parquet
    '''parquet_root = _resolve_path(cfg.get("parquet_root", "parquet"))
    dataset = ds.dataset(str(parquet_root), format="parquet")
    start_ts = pd.to_datetime(start)
    end_ts   = pd.to_datetime(end)
    filt = (
        (ds.field("timestamp") >= start_ts) &
        (ds.field("timestamp") <= end_ts) &
        (ds.field("symbol") == symbol)
    )
    table = dataset.to_table(filter=filt)
    if table.num_rows == 0:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume","symbol"])
    df = table.to_pandas()
    # Normalize types (you’ve been bitten by tz/dtype mismatches before)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["symbol"] = df["symbol"].astype("string")
    return df.sort_values("timestamp").reset_index(drop=True)
    '''

    # Fallback: hive-partitioned parquet
    parquet_root = _resolve_path(cfg.get("parquet_root", "parquet"))
    sym_dir = parquet_root / f"symbol={symbol}"
    if not sym_dir.exists():
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "symbol"])

    dataset = ds.dataset(
        str(sym_dir),
        format="parquet",
        partitioning="hive",
        exclude_invalid_files=True,
    )
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    filt = (
            (ds.field("timestamp") >= start_ts) &
            (ds.field("timestamp") <= end_ts)
    )
    table = dataset.to_table(filter=filt)
    if table.num_rows == 0:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "symbol"])
    df = table.to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["symbol"] = str(symbol)
    return df.sort_values("timestamp").reset_index(drop=True)


# ────────────────────────────────────────────────────────────────────────
# CONFIG – edit here
# ────────────────────────────────────────────────────────────────────────
CONFIG: Dict[str, Any] = {
    # raw minute-bar CSV (Date, Time, Open, High, Low, Close, Volume)
    "csv": "raw_data/RRC.csv",
    "parquet_root": "parquet",
    #"symbol": "RRC",
    "universe": StaticUniverse(["RRC", "BBY"]),
    "start": "1998-08-26",
    "end": "1999-01-01",

    "horizon_bars": 20,
    "longest_lookback_bars": 60,
    "p_gate_quantile": 0.55,
    "full_p_quantile": 0.65,
    "artifacts_root": "artifacts/a2",   # where per-fold outputs go

    # artefacts created by PathClusterEngine.build()
    "artefacts": "../weights",
    "calibration_dir": "../weights/calibration",  # <— add this

    # capital & trading costs
    "equity": 100_000.0,
    # "spread_bp": 2.0,          # half-spread in basis points
    # "commission": 0.002,       # $/share
    # "slippage_bp": 0.0,        # BrokerStub additional bp slippage
    "max_kelly": 0.5,
    "adv_cap_pct": 0.20,
    "spread_bp": 0.0,  # half-spread in basis points
    "commission": 0.0,  # $/share
    "slippage_bp": 0.0,  # BrokerStub additional bp slippage
    # debug/test toggle
    "debug_no_costs": True,  # ← set True for the tiny RRC slice
# dev gating options
    "dev_scanner_loose": True,    # ← NEW
    "dev_detector_mode": "OR",    # ← NEW; force OR even without YAML
    "sign_check": True,           # ← NEW; report will compute AUC(1−p)
    "min_entries_per_fold": 100,  # ← NEW; fail fast if < 100 entries
    # misc
    "out": "backtest_signals.csv",
    "atr_period": 14,
}


# ────────────────────────────────────────────────────────────────────────


def _resolve_path(path_like: str | Path, *, create: bool = False, is_dir: bool | None = None) -> Path:
    """
    Resolve path relative to project root if not absolute.
    If create=True, create the directory (for outputs) when it doesn't exist.
    """
    p = Path(path_like).expanduser()

    # 1) If absolute/CWD and exists → return
    if p.exists():
        return p.resolve()

    # 2) Try relative to repo root
    repo_root = Path(__file__).parents[1]
    alt = (repo_root / path_like).expanduser()
    if alt.exists():
        return alt.resolve()

    # 3) For output paths, allow creation
    if create:
        target = alt if not p.is_absolute() else p
        # If caller indicated dir, or path ends with slash, make a dir
        if is_dir or (is_dir is None and (str(target).endswith("/") or str(target).endswith("\\"))):
            target.mkdir(parents=True, exist_ok=True)
            return target.resolve()
        # Otherwise assume file path; create parents
        target.parent.mkdir(parents=True, exist_ok=True)
        return target.resolve()

    raise FileNotFoundError(f"Could not find {path_like!r} in CWD or under {repo_root!s}")


# ────────────────────────────────────────────────────────────────────────
# Portfolio aggregation helpers (Phase 4)
# ────────────────────────────────────────────────────────────────────────
def _find_symbol_outputs(artifacts_root: Path, sym: str) -> dict:
    """
    Try common file patterns produced by your backtest/walk-forward/report steps.
    Returns dict with optional keys: 'decisions', 'trades'.
    """
    root = Path(artifacts_root) / sym
    out = {}

    # decisions-like files (per-bar)
    cand_dec = []
    cand_dec += glob.glob(str(root / "decisions.parquet"))
    cand_dec += glob.glob(str(root / "signals.parquet"))
    cand_dec += glob.glob(str(root / "signals.csv"))
    cand_dec += glob.glob(str(root / "backtest_signals.csv"))
    if cand_dec:
        out["decisions"] = cand_dec[0]

    # trades/fills-like files (per-trade)
    cand_trd = []
    cand_trd += glob.glob(str(root / "trades.parquet"))
    cand_trd += glob.glob(str(root / "fills.parquet"))
    cand_trd += glob.glob(str(root / "fills.csv"))
    cand_trd += glob.glob(str(root / "trades.csv"))
    if cand_trd:
        out["trades"] = cand_trd[0]

    return out


def _read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    elif p.suffix == ".csv":
        return pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix} for {path!s}")


def _aggregate_universe_outputs(artifacts_root: Path, symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Collect per-symbol decisions & trades into two universe-wide DataFrames.
    Returns (decisions_df, trades_df) — either may be empty.
    """
    all_decisions = []
    all_trades = []

    for sym in symbols:
        found = _find_symbol_outputs(artifacts_root, sym)
        if "decisions" in found:
            df = _read_any(found["decisions"])
            df["symbol"] = sym
            all_decisions.append(df)
        if "trades" in found:
            df = _read_any(found["trades"])
            df["symbol"] = sym
            all_trades.append(df)

    dec = pd.concat(all_decisions, ignore_index=True) if all_decisions else pd.DataFrame()
    trd = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    return dec, trd



async def _run_one_symbol(sym: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout)
    log = logging.getLogger("backtest")

    # 1) Load bars for this symbol
    df_raw = _load_bars_for_symbol(cfg, sym)
    if df_raw.empty:
        raise ValueError(f"Date filter returned zero rows for symbol {sym}")

    # --- enrich raw (as you already do) ---
    df_raw["trigger_ts"] = df_raw["timestamp"]
    volume_ma = df_raw['volume'].rolling(window=20, min_periods=1).mean()
    df_raw['volume_spike_pct'] = (df_raw['volume'] / volume_ma) - 1.0
    df_raw['volume_spike_pct'] = df_raw['volume_spike_pct'].fillna(0.0)
    df_raw['prev_close'] = df_raw['close'].shift(1)

    detector = build_detectors(dev_loose=bool(cfg.get("dev_scanner_loose", False)))
    if cfg.get("dev_detector_mode", "").upper() in {"OR", "AND"}:
        detector.mode = cfg["dev_detector_mode"].upper()

    mask = await detector(df_raw)
    pass_rate = mask.mean() * 100
    print(f"[Scanner KPI] {sym}: Bars passing filters: {mask.sum()} / {len(mask)} = {pass_rate:.2f}%")

    df_full = df_raw.copy()
    df_raw = df_raw.loc[mask].reset_index(drop=True)

    df_raw["adv_shares"] = df_raw["volume"].rolling(20, min_periods=1).mean()
    df_raw["adv_dollars"] = df_raw["adv_shares"] * df_raw["close"]

    # 2) Walk-forward runner (unchanged, but pass symbol)
    #resolved_parquet_root = _resolve_path(cfg["parquet_root"])

    try:
        parquet_root_value = cfg["parquet_root"]
    except KeyError as e:
        # Make tests (and callers) get a controlled failure rather than KeyError
        raise RuntimeError("Missing required config: 'parquet_root'") from e
    resolved_parquet_root = _resolve_path(parquet_root_value)

    # --- Phase 3: ensure fresh artifacts for this symbol ---
    am = ArtifactManager(
        parquet_root=_resolve_path(cfg.get("parquet_root", "parquet")),
        artifacts_root=_resolve_path(cfg.get("artifacts_root", "artifacts/a2"), create=True, is_dir=True),
    )

    # include knobs that should force a rebuild if they change
    cfg_hash_parts = {
        "strategy": "per_symbol",
        "horizon_bars": int(cfg.get("horizon_bars", 20)),
        "longest_lookback_bars": int(cfg.get("longest_lookback_bars", 60)),
        "metric": cfg.get("metric", "mahalanobis"),
        "k_max": int(cfg.get("k_max", 64)),
        "residual_threshold": float(cfg.get("residual_threshold", 0.75)),
        "scanner_dev_loose": bool(cfg.get("dev_scanner_loose", False)),
        "detector_mode": cfg.get("dev_detector_mode", "OR"),
    }

    # Ensure artifacts/<SYM>/... are (re)built if data/knobs changed
    am.fit_or_load(
        universe=[sym],
        start=str(cfg["start"]),
        end=str(cfg["end"]),
        strategy="per_symbol",
        config_hash_parts=cfg_hash_parts,
        # per_symbol_builder=your_callable  # optional: override if you want
    )

    runner = WalkForwardRunner(
        artifacts_root=_resolve_path(cfg.get("artifacts_root", "artifacts/a2")),
        parquet_root=resolved_parquet_root,
        ev_artifacts_root=_resolve_path(cfg["artefacts"]),
        symbol=sym,
        horizon_bars=int(cfg.get("horizon_bars", 20)),
        longest_lookback_bars=int(cfg.get("longest_lookback_bars", 60)),
        p_gate_q=float(cfg.get("p_gate_quantile", 0.65)),
        full_p_q=float(cfg.get("full_p_quantile", 0.80)),
        debug_no_costs=bool(cfg.get("debug_no_costs", False)),
    )

    # metadata
    meta_path = _resolve_path(cfg.get("artifacts_root", "artifacts/a2")) / "meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    label_meta = {
        "label_horizon": f"open→open log-return, H={int(cfg['horizon_bars'])} bars",
        "label_function": "feature_engineering.labels.labeler.one_bar_ahead",
        "label_side": "long_positive",
        "label_threshold": 0.0,
        "sign_check": bool(cfg.get("sign_check", False)),
        "min_entries_per_fold": int(cfg.get("min_entries_per_fold", 0)),
        "scanner_dev_loose": bool(cfg.get("dev_scanner_loose", False)),
        "detector_mode": detector.mode,
        "use_isotonic": bool(cfg.get("use_isotonic", True)),
        "symbol": sym,
    }
    meta_doc = {**RUN_META, **label_meta}
    meta_path.write_text(json.dumps(meta_doc, indent=2))

    def _fit_isotonic_from_val(mu_val: np.ndarray, y_val: np.ndarray, *, use_iso: bool):
        if not use_iso: return None
        mu_val = np.asarray(mu_val).reshape(-1, 1)
        y_val = np.asarray(y_val).astype(float).ravel()
        if mu_val.shape[0] < 50 or len(np.unique(y_val[~np.isnan(y_val)])) < 2:
            return None
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        iso.fit(mu_val, y_val)
        return iso

    _ = runner.run(
        df_full=df_full,
        df_scanned=df_raw,
        start=cfg["start"],
        end=cfg["end"],
        calibrator_fn=lambda mu_val, y_val: _fit_isotonic_from_val(
            mu_val, y_val, use_iso=bool(cfg.get("use_isotonic", True))
        ),
        ev_engine_overrides={
            "metric": cfg.get("metric", "mahalanobis"),
            "k": int(cfg.get("k_max", 64)),
            "residual_threshold": float(cfg.get("residual_threshold", 0.75)),
        },
        persist_prob_columns=("p_raw", "p_cal"),
    )

    # per-symbol report (your script already does this)
    generate_report(
        artifacts_root=cfg.get("artifacts_root", "artifacts/a2"),
        csv_path=_resolve_path(cfg.get("csv", "raw_data")),  # not used by report, but keep signature
    )
    return pd.DataFrame([{"run_id": RUN_ID, "symbol": sym, "out_dir": str(cfg.get("artifacts_root", "artifacts/a2"))}])



# ────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────
async def run(cfg: Dict[str, Any]) -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout)
    log = logging.getLogger("backtest")

    # AFTER: log = logging.getLogger("backtest")
    if BACKTEST_MODE.upper() == "NOCLI":
        parquet_root = _resolve_path(cfg.get("parquet_root", "parquet"))
        all_syms = discover_parquet_symbols(parquet_root)
        start_ts, end_ts = _discover_time_bounds_all(parquet_root)
        cfg = {**cfg, "universe": StaticUniverse(all_syms),
               "start": start_ts.strftime("%Y-%m-%d"),
               "end": end_ts.strftime("%Y-%m-%d")}
        print(f"[Phase-3] NOCLI mode: {len(all_syms)} symbols, {cfg['start']} → {cfg['end']}")

    # Resolve universe → list[str]
    #symbols_requested = _resolve_universe(cfg.get("universe", []))

    # Resolve universe → list[str] (Step-2.2 single entry point)
    '''try:
        symbols_requested = resolve_universe(cfg, as_of=cfg.get("as_of"))
    except NotImplementedError as e:
        # Clean error if someone asks for SP500 PIT before it exists
        raise RuntimeError("SP500Universe(as_of=...) not implemented in Phase 2. Use Static/File universe.") from e
    except UniverseError as e:
        raise RuntimeError(f"Invalid universe in CONFIG: {e}") from e
    '''

    # Resolve universe → list[str] (Step-2.2 single entry point)
    try:
        symbols_requested = U.resolve_universe(cfg, as_of=cfg.get("as_of"))
    except NotImplementedError as e:
        # Clean error if someone asks for SP500 PIT before it exists
        raise RuntimeError("SP500Universe(as_of=...) not implemented in Phase 2. Use Static/File universe.") from e
    except U.UniverseError as e:
        #raise RuntimeError(f"Invalid universe in CONFIG: {e}") from e
        raise RuntimeError("Universe is empty: Resolved universe is empty")

    print(f"[Universe] size = {len(symbols_requested)} | {symbols_requested}")

    # --- Phase-2.5: provenance header -------------------------------------------
    uhash = _stable_universe_hash(symbols_requested)
    usrc = _universe_source_label(cfg)
    win_s, win_e = str(cfg["start"]), str(cfg["end"])
    arte_root = _resolve_path(cfg.get("artifacts_root", "artifacts/a2"), create=True, is_dir=True)

    print(f"[Run] universe_hash={uhash} | source={usrc} | window={win_s}→{win_e} | artifacts={arte_root}")

    #if not symbols_requested:
    #    raise ValueError("Universe is empty. Provide at least one symbol via CONFIG['universe'].")

    # --- Phase-2.6: guardrails ---------------------------------------------------
    if not symbols_requested:
        raise RuntimeError("Universe is empty. Provide at least one symbol via CONFIG['universe'].")

    max_size = int(cfg.get("universe_max_size", 10_000))
    if len(symbols_requested) > max_size:
        raise RuntimeError(
            f"Universe too large: {len(symbols_requested)} > {max_size}. "
            "Lower CONFIG['universe_max_size'] or reduce the universe."
        )

    # Optional but recommended: fast coverage precheck (zero parquet reads)
    if bool(cfg.get("strict_universe", False)):
        rep = summarize_partitions_fast(
            cfg.get("parquet_root", "parquet"),
            symbols_requested,
            cfg["start"],
            cfg["end"],
        )
        ratio = rep["coverage"]["ratio"]
        need = float(cfg.get("universe_min_coverage", 0.90))
        if ratio < need:
            have, tot = rep["coverage"]["have_data"], rep["coverage"]["total"]
            empties = rep["coverage"]["empty_symbols"]
            raise RuntimeError(
                f"Coverage {ratio * 100:.1f}% ({have}/{tot}) below required {need * 100:.0f}% "
                f"for window {cfg['start']}–{cfg['end']}. "
                f"Empty symbols: {empties}"
            )
    else:
        # Soft warning if provided
        try:
            rep = summarize_partitions_fast(
                cfg.get("parquet_root", "parquet"),
                symbols_requested,
                cfg["start"],
                cfg["end"],
            )
            ratio = rep["coverage"]["ratio"]
            need = float(cfg.get("universe_min_coverage", 0.90))
            if ratio < need:
                have, tot = rep["coverage"]["have_data"], rep["coverage"]["total"]
                empties = rep["coverage"]["empty_symbols"]
                print(f"[WARN] Coverage {ratio * 100:.1f}% ({have}/{tot}) < {need * 100:.0f}% "
                      f"for window {cfg['start']}–{cfg['end']}. Empty: {empties}")
        except Exception:
            # If manifest scan isn’t available for some reason, continue
            pass
    # ---------------------------------------------------------------------------

    # Intersect with symbols present in parquet
    project_root = Path(__file__).resolve().parents[2]  # repo root
    parquet_root = _resolve_path(cfg.get("parquet_root", "parquet"))
    available = set(discover_parquet_symbols(parquet_root))
    symbols = [s for s in symbols_requested if s in available]
    missing  = [s for s in symbols_requested if s not in available]

    # Phase-2 acceptance summary
    print("=" * 72)
    print(f"[Universe] Requested: {len(symbols_requested)} {symbols_requested[:12]}{' ...' if len(symbols_requested) > 12 else ''}")
    print(f"[Universe] Available in parquet: {len(available)}")
    if missing:
        print(f"[Universe] WARNING: missing (not under parquet/): {missing[:12]}{' ...' if len(missing) > 12 else ''}")
    print(f"[Window]  {cfg['start']} → {cfg['end']}")
    part_counts = []
    for s in symbols:
        parts = list((Path(parquet_root) / f"symbol={s}").glob("year=*/month=*"))
        part_counts.append((s, len(parts)))
    part_counts.sort(key=lambda x: x[0])
    sample = ", ".join([f"{s}:{n}" for s, n in part_counts[:12]])
    print(f"[Parquet] Partitions (symbol:count) {sample}{' ...' if len(part_counts) > 12 else ''}")
    print("=" * 72)

    if not symbols:
        raise RuntimeError("No requested symbols were found under parquet/. Abort.")

    # Loop over symbols, run one-symbol workflow
    results = []
    for sym in symbols:
        sym_cfg = {**cfg, "symbol": sym}  # pass symbol to runner
        log.info(f"=== Running backtest for {sym} ===")
        #res = await _run_one_symbol(sym, sym_cfg)
        for sym in symbols:
            try:
                res = await _run_one_symbol(sym, {**cfg, "symbol": sym})
            except RuntimeError:
                raise
            except Exception as e:
                # keep header output stable/once and fail in a controlled way
                raise RuntimeError(f"Backtest aborted for {sym}") from e

        results.append(res)





    # ─── Phase 4: Universe portfolio aggregation ────────────────────────
    arte_root = _resolve_path(cfg.get("artifacts_root", "artifacts/a2"), create=True, is_dir=True)

    dec_df, trd_df = _aggregate_universe_outputs(arte_root, symbols)

    # Make a portfolio folder
    port_dir = arte_root / "portfolio"
    port_dir.mkdir(parents=True, exist_ok=True)

    if not dec_df.empty:
        # Save unified decisions (per-bar across symbols)
        # Expect at least: ['timestamp','p_raw'/'p_cal', ... , 'symbol']
        # We'll not enforce schema here; this is a pass-through aggregator.
        dec_out = port_dir / "decisions.parquet"
        dec_df.to_parquet(dec_out, index=False)
        print(f"[Portfolio] decisions → {dec_out}")

    if not trd_df.empty:
        # Normalize minimal expected schema for curve:
        #  prefer columns: ['entry_ts','exit_ts','entry_price','exit_price','realized_pnl', 'symbol']
        # If your fills have different names, map them here:
        colmap = {}
        if "pnl" in trd_df.columns and "realized_pnl" not in trd_df.columns:
            colmap["pnl"] = "realized_pnl"
        if "exit_time" in trd_df.columns and "exit_ts" not in trd_df.columns:
            colmap["exit_time"] = "exit_ts"
        if colmap:
            trd_df = trd_df.rename(columns=colmap)

        # Save unified trades
        trades_out = port_dir / "trades.parquet"
        trd_df.to_parquet(trades_out, index=False)
        print(f"[Portfolio] trades → {trades_out}")

        # Build a simple realized-PnL equity curve (no open PnL)
        if "exit_ts" in trd_df.columns and "realized_pnl" in trd_df.columns:
            curve = equity_curve_from_trades(trd_df)
            curve_out = port_dir / "equity_curve.csv"
            curve.to_csv(curve_out, header=["equity"])
            print(f"[Portfolio] equity curve → {curve_out}")
        else:
            print("[Portfolio] skipped equity curve (missing exit_ts or realized_pnl)")

    # ─── end Phase 4 ────────────────────────────────────────────────────


    # --- Phase-4: consolidate per-fold/per-symbol outputs into single files ---
    artifacts_root = _resolve_path(cfg.get("artifacts_root", "artifacts/a2"), create=True, is_dir=True)
    dec_path, trd_path = _consolidate_phase4_outputs(artifacts_root)

    # Soft acceptance checks: only probe if the files exist
    if dec_path is not None and trd_path is not None:
        dec = pd.read_parquet(dec_path)
        trd = pd.read_parquet(trd_path)

        print("decisions rows:", len(dec), "trades rows:", len(trd))
        print("symbols in decisions:", sorted(dec['symbol'].dropna().astype(str).unique().tolist()) if 'symbol' in dec.columns else "(no symbol column)")
        print("symbols in trades:",    sorted(trd['symbol'].dropna().astype(str).unique().tolist()) if 'symbol' in trd.columns else "(no symbol column)")

        # unified clock sanity: timestamps with >=2 symbols
        if {'timestamp','symbol'}.issubset(dec.columns):
            multi = (dec.groupby('timestamp')['symbol'].nunique() >= 2).mean()
            print("share of timestamps with >=2 symbols present:", round(float(multi), 3))

        # optional: decision→entry causality check (if you persist these)
        if {'decision_ts','entry_ts'}.issubset(dec.columns):
            lag_ok = (pd.to_datetime(dec['entry_ts']) > pd.to_datetime(dec['decision_ts'])).mean()
            print("entry strictly after decision:", round(float(lag_ok), 3))
    else:
        print("[Phase-4] Note: could not find per-fold outputs to consolidate into decisions/trades. "
              "Backtest completed, but portfolio acceptance checks were skipped.")



    # decisions must exist and contain both symbols on a shared timeline
    #import pandas as pd
    '''dec = pd.read_parquet("artifacts/a2/decisions.parquet")
    trd = pd.read_parquet("artifacts/a2/trades.parquet")

    print("decisions rows:", len(dec), "trades rows:", len(trd))
    print("symbols in decisions:", dec['symbol'].unique())
    print("symbols in trades:", trd['symbol'].unique())

    # unified clock sanity: at least some timestamps with >1 symbol
    multi = (dec.groupby('timestamp')['symbol'].nunique() > 1).mean()
    print("share of timestamps with >=2 symbols present:", round(multi, 3))

    # overlapping positions → portfolio behavior (requires portfolio equity in trades or a portfolio ledger)
    have_equity_cols = {'equity', 'cash', 'gross', 'net'}.issubset(trd.columns)
    print("trades has portfolio columns:", have_equity_cols)

    # no look-ahead: entries should be at t+1 vs the decision bar time
    if {'decision_ts', 'entry_ts'}.issubset(dec.columns):
        lag_ok = ((pd.to_datetime(dec['entry_ts']) > pd.to_datetime(dec['decision_ts'])).mean())
        print("entry strictly after decision:", round(lag_ok, 3))
'''
    # Concatenate tiny summaries
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()



__all__ = ["run", "CONFIG"]

if __name__ == "__main__":
    asyncio.run(run(CONFIG))