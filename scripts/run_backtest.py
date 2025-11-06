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
from prediction_engine.portfolio.sizer import size_from_p, RiskCaps
# NEW (Phase 4.5): portfolio ledger + risk
from prediction_engine.portfolio.ledger import PortfolioLedger, TradeFill
from prediction_engine.portfolio.risk import RiskLimits, RiskEngine

# NEW (Phase 4.5): portfolio ledger + risk
from prediction_engine.portfolio.ledger import PortfolioLedger, TradeFill
from prediction_engine.portfolio.risk import RiskLimits, RiskEngine


import logging
import sys
from pathlib import Path



# at top of scripts/run_backtest.py
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.types as pat

def _arrow_ts_between_filter(dataset: ds.Dataset, col: str, start: str, end: str):
    """
    Build an Arrow filter `(col >= start) & (col <= end)` where the Python datetime
    scalars are cast to the SAME timestamp(unit, tz) as the dataset column.
    Works for tz-aware (UTC) and tz-naive columns, any unit (s/ms/us/ns).
    """
    tfield = dataset.schema.field(col).type
    if not pat.is_timestamp(tfield):
        raise TypeError(f"Column {col!r} is not a timestamp: {tfield}")

    unit = tfield.unit  # 's'|'ms'|'us'|'ns'
    tz = tfield.tz      # None or 'UTC' (or other tz)

    # Build Python datetimes with matching tz-awareness
    if tz is None:
        s_py = pd.to_datetime(start).to_pydatetime()
        e_py = pd.to_datetime(end).to_pydatetime()
    else:
        s_py = pd.to_datetime(start, utc=True).to_pydatetime()
        e_py = pd.to_datetime(end,   utc=True).to_pydatetime()

    s_scalar = pa.scalar(s_py, type=pa.timestamp(unit, tz))
    e_scalar = pa.scalar(e_py, type=pa.timestamp(unit, tz))

    col_expr = ds.field(col)
    return (col_expr >= s_scalar) & (col_expr <= e_scalar)




# --- Phase 4.1: unified minute clock over the universe (fast, no joins) ---
def build_unified_clock(parquet_root: Path | str, start: str, end: str, symbols: list[str]) -> pd.DatetimeIndex:
    root = Path(parquet_root)
    parts: list[pd.Series] = []

    for sym in symbols:
        sym_dir = root / f"symbol={sym}"
        if not sym_dir.exists():
            continue

        dataset = ds.dataset(str(sym_dir), format="parquet", partitioning="hive", exclude_invalid_files=True)
        filt = _arrow_ts_between_filter(dataset, "timestamp", start, end)
        tbl = dataset.to_table(columns=["timestamp"], filter=filt)
        if tbl.num_rows == 0:
            continue

        # Whatever the on-disk type is, normalize to tz-aware UTC in pandas
        ts = pd.to_datetime(tbl.column("timestamp").to_pandas(), utc=True, errors="coerce")
        if not ts.empty:
            parts.append(ts.dropna())

    if not parts:
        return pd.DatetimeIndex([], tz="UTC")

    uni = pd.Index(pd.concat(parts, ignore_index=True).unique())
    uni = pd.to_datetime(uni, utc=True).floor("T").sort_values().unique()
    return pd.DatetimeIndex(uni, tz="UTC")




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

# after: def _pca_cols(df: pd.DataFrame) -> list[str]:
from prediction_engine.scoring.batch import vectorize_minute_batch  # NEW (Step 4.6)


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
    #filt = (
    #        (ds.field("timestamp") >= start_ts) &
    #        (ds.field("timestamp") <= end_ts)
    #)
    #table = dataset.to_table(filter=filt)
    #if table.num_rows == 0:
    #    return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "symbol"])
    '''df = table.to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["symbol"] = str(symbol)'''

    #df = table.to_pandas()
    #df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # ▼ NEW: enforce unique timestamps after IO and any tz fiddling
    #df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

    #df["symbol"] = str(symbol)
    #return df.reset_index(drop=True)
    filt = _arrow_ts_between_filter(dataset, "timestamp", cfg["start"], cfg["end"])
    table = dataset.to_table(filter=filt)
    if table.num_rows == 0:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "symbol"])

    df = table.to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # ensure unique, ordered minutes (helps later searchsorted calls)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    df["symbol"] = str(symbol)
    return df.reset_index(drop=True)
    #return df.sort_values("timestamp").reset_index(drop=True)



# --- Phase 4.1 helper: get the feature row for (symbol, t) without labels ---
def _slice_features_at(df_features: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """
    Return a single-row DataFrame at minute t (UTC). Caller is responsible for
    ensuring df_features['timestamp'] is tz-aware UTC. No label look-ahead here.
    """
    t = pd.to_datetime(t, utc=True)
    # Exact match on minute; if your FE emits seconds, floor to minute first
    view = df_features.loc[df_features["timestamp"].dt.floor("T") == t]
    # We intentionally do not compute or touch any H-ahead targets here.
    return view.head(1)


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
    #"spread_bp": 0.0,  # half-spread in basis points
    #"commission": 0.0,  # $/share
    #"slippage_bp": 0.0,  # BrokerStub additional bp slippage
    # debug/test toggle
    #"debug_no_costs": True,  # ← set True for the tiny RRC slice
    "vectorized_scoring": True,  # Step 4.6: enable batch scoring by minute across symbols


    # Costs ON by default — disable only for micro unit tests
    "commission": 0.005,      # $/share (e.g., $0.005)
    "debug_no_costs": False,  # True only for tiny synthetic tests
    # optional: if you later record half-spread per trade, we'll use that directly


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


# --- Phase 4.2: attach modeled costs to trades (commission + spread + impact)
def _apply_modeled_costs_to_trades(trades: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Add round-trip modeled costs and net PnL:
      modeled_cost_total = spread + commission + slippage + market_impact

    Sane defaults so tests never get NaNs:
      • if half_spread_usd missing ⇒ derive from cfg['spread_bp'] (default 1.0 bp)
      • if adv_frac missing ⇒ 0.0
      • commission/slippage default to 0.0
      • market impact default: impact_bps_per_adv_frac (default 25 bps per 1.0 ADV fraction)

    Returns a copy with columns:
      'modeled_cost_total', 'realized_pnl_after_costs'
    """
    df = trades.copy()

    # Short-circuit for costless mode
    if bool(cfg.get("debug_no_costs", False)):
        df["modeled_cost_total"] = 0.0
        df["realized_pnl_after_costs"] = df["realized_pnl"].astype(float)
        return df

    # Pull knobs with safe defaults
    spread_bp = float(cfg.get("spread_bp", 1.0))          # half-spread *in bps of price*
    commission_per_share = float(cfg.get("commission", 0.0))
    slippage_bp = float(cfg.get("slippage_bp", 0.0))
    impact_bps_per_frac = float(cfg.get("impact_bps_per_adv_frac", 25.0))

    # Required numeric columns with safe types
    qty = df.get("qty", 0).astype(float).abs()
    entry = df.get("entry_price", 0).astype(float)
    exit_ = df.get("exit_price", 0).astype(float)
    mid = (entry.add(exit_)).div(2.0)

    # half_spread_usd: use provided, else derive from bps*price
    half_spread_usd = df.get("half_spread_usd")
    if half_spread_usd is None:
        half_spread_usd = pd.Series(index=df.index, dtype=float)
    half_spread_usd = half_spread_usd.astype(float)
    fallback_half_spread = (spread_bp / 1e4) * mid  # USD/share
    half_spread_usd = half_spread_usd.where(~half_spread_usd.isna(), fallback_half_spread)

    # adv_frac (for market impact) — default 0
    adv_frac = df.get("adv_frac")
    if adv_frac is None:
        adv_frac = pd.Series(0.0, index=df.index)
    adv_frac = adv_frac.astype(float).fillna(0.0)

    # Components (all positive)
    spread_cost = 2.0 * half_spread_usd * qty
    commission_cost = 2.0 * commission_per_share * qty
    slippage_cost = (slippage_bp / 1e4) * mid * 2.0 * qty
    impact_bp = impact_bps_per_frac * adv_frac           # bps
    impact_cost = (impact_bp / 1e4) * mid * qty

    total = (spread_cost + commission_cost + slippage_cost + impact_cost).fillna(0.0)

    df["modeled_cost_total"] = total
    df["realized_pnl_after_costs"] = df["realized_pnl"].astype(float) - total
    return df

# Phase 4.3 – simulator & quotes
from prediction_engine.portfolio.order_sim import simulate_entry, simulate_exit, QuoteStats
from prediction_engine.portfolio.quotes import estimate_quote_stats_from_rolling

# --- Phase 4.3: decisions → simulated trades (deterministic MOO/MOC, partial fills)
def _simulate_trades_from_decisions(
    decisions: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    rules: dict,
    horizon_col: str = "horizon_bars",
    target_qty_col: str = "target_qty",
) -> pd.DataFrame:
    """
    Turn per-bar 'decisions' into trades with entry/exit at next opens.
    Requirements:
      decisions: ['timestamp','symbol', target_qty_col, horizon_col, ...]
      bars:      ['timestamp','symbol','open','volume', ('adv_shares' optional)]
    Returns DataFrame with: ['symbol','entry_ts','entry_price','exit_ts','exit_price',
                             'bars_held','qty','realized_pnl','half_spread_usd','adv_frac']
    """
    if decisions.empty:
        return pd.DataFrame()

    import pandas as _pd
    dec = decisions.copy()
    dec["timestamp"] = _pd.to_datetime(dec["timestamp"], utc=True)
    bars = bars.copy()
    bars["timestamp"] = _pd.to_datetime(bars["timestamp"], utc=True)

    # Build next-open lookup per symbol
    bars_by_sym = {s: g.sort_values("timestamp").reset_index(drop=True) for s, g in bars.groupby("symbol")}
    def _next_open(sym: str, t):
        g = bars_by_sym.get(sym)
        if g is None: return None, None, None
        idx = g["timestamp"].searchsorted(t, side="right")  # next bar
        if idx >= len(g): return None, None, None
        return float(g.loc[idx, "open"]), _pd.Timestamp(g.loc[idx, "timestamp"]), float(g.loc[idx, "volume"])

    rows = []
    for i, r in dec.iterrows():
        sym = str(r["symbol"])
        tgt = float(r.get(target_qty_col, 0.0))
        H   = int(r.get(horizon_col, 1))
        if tgt == 0:
            continue

        # entry at next open
        px_e, ts_e, vol_e = _next_open(sym, r["timestamp"])
        if ts_e is None:
            continue

        # crude quote stats from bars (spread & ADV)
        q = estimate_quote_stats_from_rolling(
            {"open": px_e, "adv_shares": bars_by_sym[sym]["volume"].rolling(20, min_periods=1).mean().iloc[max(0, bars_by_sym[sym]["timestamp"].searchsorted(ts_e)-1)]}
        )

        ent = simulate_entry(
            decision_row=_pd.Series({"target_qty": tgt, "bar_volume": vol_e}),
            next_open_price=px_e,
            next_open_ts=ts_e,
            quote=q,
            rules=rules,
        )

        # exit H bars later at open
        # index of entry bar in symbol series:
        g = bars_by_sym[sym]
        idx_e = int(g["timestamp"].searchsorted(ts_e))
        idx_x = idx_e + H
        if idx_x >= len(g):
            continue
        px_x = float(g.loc[idx_x, "open"]); ts_x = _pd.Timestamp(g.loc[idx_x, "timestamp"])
        ex = simulate_exit(
            position_row=_pd.Series({"filled_qty": ent.filled_qty, "entry_price": ent.entry_price}),
            exit_open_price=px_x,
            exit_open_ts=ts_x,
            bars_held=H,
            quote=q,
            rules=rules,
        )

        rows.append({
            "symbol": sym,
            "entry_ts": ent.entry_ts,
            "entry_price": ent.entry_price,
            "exit_ts": ex.exit_ts,
            "exit_price": ex.exit_price,
            "bars_held": ex.bars_held,
            "qty": ent.filled_qty,
            "realized_pnl": ex.realized_pnl,
            "half_spread_usd": ent.half_spread_usd,
            "adv_frac": ent.adv_frac,
        })

    return _pd.DataFrame(rows)



# --- Phase 4.4: compute target_qty from calibrated p with a dose–response ramp + cost hurdle
# --- Phase 4.4: compute target_qty from calibrated p with a dose–response ramp + cost hurdle
def _apply_sizer_to_decisions(
    decisions: pd.DataFrame,
    bars: pd.DataFrame,
    cfg: dict,
    *,
    target_qty_col: str = "target_qty",   # <- define it here
) -> pd.DataFrame:
    """
    Fills `target_qty_col` in decisions using size_from_p on the next-open price and simple vol proxy.
    Requirements:
      decisions: ['timestamp','symbol','p_cal','horizon_bars', ...]
      bars:      ['timestamp','symbol','open','volume'] (for next-open price and ADV/vol proxies)
    """
    import pandas as _pd
    import numpy as np

    if decisions.empty:
        return decisions

    dec = decisions.copy()
    dec["timestamp"] = _pd.to_datetime(dec["timestamp"], utc=True)
    bars = bars.copy()
    bars["timestamp"] = _pd.to_datetime(bars["timestamp"], utc=True)

    # Build per-symbol series
    bars_by_sym = {s: g.sort_values("timestamp").reset_index(drop=True) for s, g in bars.groupby("symbol")}
    # Rolling 20-bar sigma of log returns (per-symbol) as vol proxy
    for s, g in bars_by_sym.items():
        rets = _pd.Series(_pd.Series(g["open"]).astype(float)).pct_change().fillna(0.0)
        g["sigma20"] = rets.rolling(20, min_periods=5).std().fillna(rets.std() or 0.005)
        g["adv_shares"] = g["volume"].rolling(20, min_periods=1).mean().fillna(g["volume"])
        bars_by_sym[s] = g

    caps = RiskCaps(
        max_gross_frac=float(cfg.get("max_gross_frac", 0.10)),
        adv_cap_pct=float(cfg.get("adv_cap_pct", 0.20)),
        max_shares=float(cfg.get("max_shares", 1e9)),
    )
    p_gate = float(cfg.get("p_gate_quantile", 0.55))
    p_full = float(cfg.get("full_p_quantile", 0.65))
    capital = float(cfg.get("equity", 100_000.0))
    commission = float(cfg.get("commission", 0.0))
    slippage_bp = float(cfg.get("slippage_bp", 0.0))
    impact_bps_per_frac = float(cfg.get("impact_bps_per_adv_frac", 25.0))
    cost_lambda = float(cfg.get("sizer_cost_lambda", 1.2))
    strategy = cfg.get("sizer_strategy", "score")

    target_qty = []
    for _, r in dec.iterrows():
        sym = str(r["symbol"])
        g = bars_by_sym.get(sym)
        if g is None or "p_cal" not in r or _pd.isna(r["p_cal"]):
            target_qty.append(0.0)
            continue

        # look up next-open for price + contemporaneous vol/ADV
        idx = g["timestamp"].searchsorted(r["timestamp"], side="right")
        if idx >= len(g):
            target_qty.append(0.0)
            continue

        price = float(g.loc[idx, "open"])
        vol = float(g.loc[max(0, idx - 1), "sigma20"])
        adv = float(g.loc[max(0, idx - 1), "adv_shares"])
        half_spread_usd = float(cfg.get("half_spread_usd", (cfg.get("spread_bp", 2.0) / 1e4) * price))

        costs = {
            "price": price,
            "half_spread_usd": half_spread_usd,
            "commission": commission,
            "slippage_bp": slippage_bp,
            "impact_bps_per_adv_frac": impact_bps_per_frac,
            "adv_shares": adv,
            "adv_frac": 0.0,
        }

        qty = size_from_p(
            float(r["p_cal"]),
            vol=vol if vol > 0 else float(cfg.get("fallback_vol", 0.005)),
            capital=capital,
            risk_caps=caps,
            costs=costs,
            p_gate=p_gate,
            p_full=p_full,
            strategy=strategy,
            cost_lambda=cost_lambda,
        )

        # Fallback ramp if sized 0 but above gate
        if (qty is None) or (qty == 0.0 and float(r["p_cal"]) > p_gate):
            ramp = 0.0
            if p_full > p_gate:
                ramp = (float(r["p_cal"]) - p_gate) / (p_full - p_gate)
            ramp = float(np.clip(ramp, 0.0, 1.0))
            max_dollars = caps.max_gross_frac * capital
            max_shares_gross = max_dollars / max(price, 1e-8)
            soft_cap_shares = min(max_shares_gross, caps.max_shares)
            qty = float(max(0.0, ramp * soft_cap_shares))
            if 0.0 < qty < 1.0:
                qty = 1.0

        target_qty.append(qty)

    dec[target_qty_col] = target_qty
    return dec



# --- Phase 4.5: risk gating before turning decisions into trades -------------
def _enforce_risk_on_decisions(decisions: pd.DataFrame,
                               bars: pd.DataFrame,
                               risk: RiskEngine,
                               ledger: PortfolioLedger,
                               *, symbol_sector: dict[str, str] | None = None,
                               log=None) -> pd.DataFrame:
    """
    Drop/clip decisions that would violate risk limits at the moment of entry.
    Approximates entry price with next-open (same lookup as the simulator).
    """
    import pandas as _pd
    if decisions.empty:
        return decisions

    dec = decisions.copy()
    dec["timestamp"] = _pd.to_datetime(dec["timestamp"], utc=True)
    bars = bars.copy()
    bars["timestamp"] = _pd.to_datetime(bars["timestamp"], utc=True)

    # Build per symbol next-open lookup
    bars_by_sym = {s: g.sort_values("timestamp").reset_index(drop=True) for s, g in bars.groupby("symbol")}

    # Base snapshot from ledger (will be cloned and updated locally per timestamp)
    def _base_snapshot():
        return {
            **ledger.snapshot_row(),
            "positions": [s for s, p in ledger.positions.items() if p.qty != 0],
            "symbol_notional": {s: abs(p.qty * p.avg_price) for s, p in ledger.positions.items()},
            "sector_gross": {},  # local working view
            "open_positions": sum(1 for p in ledger.positions.values() if p.qty != 0),
        }

    # Seed sector view from existing positions (approx using avg_price)
    if hasattr(risk, "sector_gross") and isinstance(risk.sector_gross, dict):
        sector_seed = dict(risk.sector_gross)
    else:
        sector_seed = {}

    kept_rows = []

    # Process decisions grouped by timestamp so we can enforce max_concurrent "within the bar"
    for tstamp, group in dec.sort_values(["timestamp"]).groupby("timestamp"):
        snap = _base_snapshot()
        # start local view with risk engine's sector gross
        snap["sector_gross"] = dict(sector_seed)

        # loop deterministic by symbol to keep behavior stable
        for _, r in group.sort_values(["symbol"]).iterrows():
            sym = str(r["symbol"])
            g = bars_by_sym.get(sym)
            if g is None:
                continue
            idx = g["timestamp"].searchsorted(tstamp, side="right")
            if idx >= len(g):
                continue
            price = float(g.loc[idx, "open"])
            qty = float(r.get("target_qty", 0.0))
            if qty == 0.0:
                continue

            sector = (symbol_sector or {}).get(sym, None)
            ok, reason = risk.can_open(
                symbol=sym, sector=sector, qty=qty, price=price, now=g.loc[idx, "timestamp"],
                ledger_snapshot={
                    "gross": snap["gross"],
                    "net": snap["net"],
                    "day_pnl": snap["day_pnl"],
                    "positions": snap["positions"],
                    "symbol_notional": snap["symbol_notional"],
                },
                open_positions=snap["open_positions"],
            )

            if ok:
                # Accept and update LOCAL snapshot so later decisions at this same timestamp
                # see the increased exposure/concurrency.
                kept_rows.append(r)
                add_notional = abs(qty * price)
                side = 1 if qty > 0 else -1
                snap["gross"] = snap["gross"] + add_notional
                snap["net"] = snap["net"] + side * add_notional
                snap["open_positions"] = snap["open_positions"] + (0 if sym in snap["positions"] else 1)
                snap["positions"] = list(set(snap["positions"]) | {sym})
                snap["symbol_notional"][sym] = snap["symbol_notional"].get(sym, 0.0) + add_notional
                if sector:
                    snap["sector_gross"][sector] = snap["sector_gross"].get(sector, 0.0) + add_notional
            elif log:
                log.info(f"[Risk] skip {sym} qty={qty:.0f} @ {price:.2f} — {reason}")

    return _pd.DataFrame(kept_rows, columns=dec.columns) if kept_rows else dec.iloc[:0]


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

    # --- Test-only override: force constant p to exercise the full Phase-4 pipeline on bare OHLCV hives ---
    unit_p = float(cfg.get("unit_test_force_constant_p", float("nan")))
    if unit_p == unit_p:  # not-NaN → enabled
        # Build minimal per-bar decisions (skip the very last bar; we need t+1 to exist)
        dec = df_raw.loc[df_raw["timestamp"] < df_raw["timestamp"].max()].copy()
        dec = dec[["timestamp"]].copy()
        dec["symbol"] = str(sym)
        dec["p_raw"] = unit_p
        dec["p_cal"] = unit_p
        dec["horizon_bars"] = int(cfg.get("horizon_bars", 3))

        # Sizer → target_qty on next-open
        bars_min = df_full[["timestamp","symbol","open","volume"]].copy()
        sized = _apply_sizer_to_decisions(decisions=dec, bars=bars_min, cfg=cfg)
        sized = sized.loc[sized["target_qty"].abs() > 0].copy()

        # Simulate fills (MOO/MOC with gap bands + partial fills)
        rules = {
            "max_participation": float(cfg.get("max_participation", 0.25)),
            "moo_gap_band": True, "moc_gap_band": True,
        }
        trades = _simulate_trades_from_decisions(
            decisions=sized, bars=bars_min, rules=rules,
            horizon_col="horizon_bars", target_qty_col="target_qty",
        )

        # Persist per-symbol artifacts so the Phase-4.7 aggregator can pick them up
        sym_dir = _resolve_path(cfg.get("artifacts_root", "artifacts/a2"), create=True, is_dir=True) / str(sym)
        sym_dir.mkdir(parents=True, exist_ok=True)
        if len(sized):
            sized.to_parquet(sym_dir / "decisions.parquet", index=False)
        if len(trades):
            trades.to_parquet(sym_dir / "trades.parquet", index=False)

        # Short-circuit the per-symbol runner in this test mode; aggregation runs later in `run()`
        return pd.DataFrame([{"run_id": RUN_ID, "symbol": sym, "out_dir": str(sym_dir)}])


    # 2) Walk-forward runner (unchanged, but pass symbol)
    resolved_parquet_root = _resolve_path(cfg["parquet_root"])

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
        # INSERT INTO cfg_hash_parts (same dict), AFTER detector_mode:
        "regime_settings": cfg.get("regime_settings", {"modes": ["TREND", "RANGE", "VOL"]}),
        "gating_flags": {
            "p_gate_quantile": float(cfg.get("p_gate_quantile", 0.55)),
            "full_p_quantile": float(cfg.get("full_p_quantile", 0.65)),
            "sign_check": bool(cfg.get("sign_check", False)),
        },
        "distance_family": cfg.get("metric", "mahalanobis"),

    }

    # INSERT AFTER cfg_hash_parts definition
    schema_hash_parts = {
        # Feature schema identity: either a version tag or explicit feature list.
        # If you have a concrete list, pass it in config as 'feature_schema_list'.
        "feature_schema_version": cfg.get("feature_schema_version", "v1"),
        "feature_schema_list": cfg.get("feature_schema_list", []),

        # Label identity: horizon & function
        "label_horizon_bars": int(cfg.get("horizon_bars", 20)),
        "label_function": "open_to_open_log_return",  # keep stable string id used across runs

        # Distance family (affects neighbor geometry)
        "distance_family": cfg.get("metric", "mahalanobis"),
        # Regime settings and gating flags (impact template selection)
        "regime_settings": cfg.get("regime_settings", {"modes": ["TREND", "RANGE", "VOL"]}),
        "gating_flags": {
            "p_gate_quantile": float(cfg.get("p_gate_quantile", 0.55)),
            "full_p_quantile": float(cfg.get("full_p_quantile", 0.65)),
            "sign_check": bool(cfg.get("sign_check", False)),
        },
    }

    # Ensure artifacts/<SYM>/... are (re)built if data/knobs changed
    '''am.fit_or_load(
        universe=[sym],
        start=str(cfg["start"]),
        end=str(cfg["end"]),
        strategy="per_symbol",
        config_hash_parts=cfg_hash_parts,
        # per_symbol_builder=your_callable  # optional: override if you want
        # INSERT INSIDE am.fit_or_load(...) ARGUMENTS, AFTER config_hash_parts=cfg_hash_parts,
        schema_hash_parts=schema_hash_parts,

    )'''

    # Example of passing builders
    from scripts.rebuild_artefacts import build_pooled_core, fit_symbol_calibrator
    am.fit_or_load(
        universe=[sym], start=cfg["start"], end=cfg["end"],
        strategy="pooled",
        config_hash_parts=cfg_hash_parts,
        schema_hash_parts=schema_hash_parts,
        pooled_builder=lambda syms, out_dir, s, e: build_pooled_core(syms, out_dir, s, e, am.parquet_root,
                                                                     n_clusters=int(cfg.get("k_max", 64))),
        calibrator_builder=lambda sym, pooled_dir, s, e: fit_symbol_calibrator(sym, pooled_dir, s, e, am.parquet_root),
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


def _score_minute_batch_shim(ev_engine, minute_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transitional shim for Step 4.6: if df has PCA feature columns, call a single
    vectorized predict. Otherwise, return empty and let the legacy path run.
    This keeps the wiring safe while you progressively feed feature rows here.
    """
    if minute_df.empty:
        return minute_df
    pca_cols = [c for c in minute_df.columns if c.startswith("pca_")]
    if not pca_cols:
        return pd.DataFrame(columns=["timestamp","symbol","p_raw","p_cal"])
    res = vectorize_minute_batch(ev_engine, minute_df, pca_cols)
    return res.frame


# ─── Step 4.8 helpers (module scope) ───────────────────────────────────

def _share_multisymbol(decisions: pd.DataFrame) -> float:
    if decisions.empty or not {'timestamp', 'symbol'}.issubset(decisions.columns):
        return 0.0
    g = decisions.groupby('timestamp')['symbol'].nunique()
    return float((g >= 2).mean())

def _median_slippage_error_bps(trades: pd.DataFrame) -> float:
    """
    Approx 'replay' check: when MOO/MOC simulator wrote half_spread_usd, the ideal
    execution is open ± half_spread. Compare what we actually recorded vs ideal.
    """
    if trades.empty:
        return 0.0
    df = trades.copy()
    for col in ("entry_price", "exit_price", "half_spread_usd"):
        if col not in df.columns:
            return 0.0

    # +1 for long, -1 for short if qty present; default +1
    side = pd.Series(1.0, index=df.index)
    if "qty" in df.columns:
        side = (df["qty"] > 0).astype(float).where(df["qty"].notna(), 1.0) * 2.0 - 1.0

    ideal_entry = df["entry_price"] - (side * df["half_spread_usd"] * -1.0)
    ideal_exit  = df["exit_price"]  + (side * df["half_spread_usd"] * -1.0)

    def _bps(err, base):
        base = base.where(base.abs() > 1e-12, 1.0)
        return (err.abs() / base) * 1e4

    entry_bps = _bps((df["entry_price"] - ideal_entry), df["entry_price"])
    exit_bps  = _bps((df["exit_price"]  - ideal_exit),  df["exit_price"])
    both = pd.concat([entry_bps, exit_bps], ignore_index=True)
    return float(both.median(skipna=True))

def _sizing_sanity(decisions: pd.DataFrame, *, p_threshold: float = 0.60, p_gate: float | None = None) -> dict:
    """
    (a) median target size for p>=threshold > 0
    (b) count 'cost-hurdle violations' where p >= gate but target_qty==0
    """
    out = {"median_pos_gt_zero": False, "cost_hurdle_violations": 0}
    if decisions.empty:
        return out
    if "p_cal" not in decisions.columns or "target_qty" not in decisions.columns:
        return out

    hi = decisions.loc[decisions["p_cal"] >= p_threshold]
    if len(hi):
        out["median_pos_gt_zero"] = float(hi["target_qty"].fillna(0).abs().median()) > 0.0

    gate = float(p_gate) if p_gate is not None else p_threshold
    #v = decisions.loc[(decisions["p_cal"] >= gate) & (decisions["target_qty"].fillna(0) == 0)]
    # Strictly above the gate is a violation if size is still zero.
    eps = 1e-12
    v = decisions.loc[(decisions["p_cal"] > gate + eps) & (decisions["target_qty"].fillna(0) == 0)]

    out["cost_hurdle_violations"] = int(len(v))
    return out

def _promotion_checks_step4_8(*, port_dir: Path, cfg: dict | None = None) -> dict:
    """
    Reads portfolio/decisions.parquet & trades.parquet, evaluates readiness bar.
    Returns dict with booleans + key metrics; prints human-readable summary.
    """
    dec_p = port_dir / "decisions.parquet"
    trd_p = port_dir / "trades.parquet"
    results = {
        "exists_decisions": dec_p.exists(),
        "exists_trades": trd_p.exists(),
        "exists_equity": (port_dir / "equity_curve.csv").exists(),
        "exists_metrics_json": (port_dir / "portfolio_metrics.json").exists(),
        "share_multisymbol": 0.0,
        "slippage_median_bps": 0.0,
        "commission_nonzero": False,
        "sizing_median_pos_gt0": False,
        "sizing_cost_hurdle_violations": None,
        "passed": False,
    }
    if not (results["exists_decisions"] and results["exists_trades"]):
        print("[4.8] Missing portfolio decisions/trades; cannot evaluate promotion.")
        return results

    decisions = pd.read_parquet(dec_p)
    trades = pd.read_parquet(trd_p)

    # Multi-symbol presence
    results["share_multisymbol"] = _share_multisymbol(decisions)

    # PnL realism
    results["slippage_median_bps"] = _median_slippage_error_bps(trades)
    if "modeled_cost_total" in trades.columns:
        if cfg and float(cfg.get("commission", 0.0)) > 0.0:
            results["commission_nonzero"] = True
        else:
            results["commission_nonzero"] = bool((trades["modeled_cost_total"] > 0).any())

    # Sizing sanity
    p_gate = float(cfg.get("p_gate_quantile", 0.55)) if cfg else None
    sz = _sizing_sanity(decisions, p_threshold=0.60, p_gate=p_gate)
    results["sizing_median_pos_gt0"] = bool(sz["median_pos_gt_zero"])
    results["sizing_cost_hurdle_violations"] = int(sz["cost_hurdle_violations"])

    # Pass/fail
    checks = [
        (results["share_multisymbol"] >= 0.10, "multi-symbol share ≥ 0.10"),
        (results["slippage_median_bps"] < 10.0, "slippage median < 10 bps"),
        (results["commission_nonzero"], "commission line non-zero"),
        (results["sizing_median_pos_gt0"], "median size > 0 for p≥0.60"),
        (results["sizing_cost_hurdle_violations"] == 0, "0% cost-hurdle violations"),
        (results["exists_equity"] and results["exists_metrics_json"], "equity+metrics emitted"),
    ]
    results["passed"] = all(ok for ok, _ in checks)

    print("[4.8] Promotion checklist:")
    for ok, label in checks:
        print(f"    [{'OK' if ok else 'FAIL'}] {label}")
    print(f"    share_multisymbol={results['share_multisymbol']:.3f}  "
          f"slip_median_bps={results['slippage_median_bps']:.2f}  "
          f"sizing_cost_hurdle_violations={results['sizing_cost_hurdle_violations']}")
    return results




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
        raise RuntimeError(f"Invalid universe in CONFIG: {e}") from e

    print(f"[Universe] size = {len(symbols_requested)} | {symbols_requested}")

    # --- Phase-2.5: provenance header -------------------------------------------
    uhash = _stable_universe_hash(symbols_requested)
    usrc = _universe_source_label(cfg)
    win_s, win_e = str(cfg["start"]), str(cfg["end"])
    arte_root = _resolve_path(cfg.get("artifacts_root", "artifacts/a2"), create=True, is_dir=True)

    print(f"[Run] universe_hash={uhash} | source={usrc} | window={win_s}→{win_e} | artifacts={arte_root}")

    if not symbols_requested:
        raise ValueError("Universe is empty. Provide at least one symbol via CONFIG['universe'].")

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


    # Phase 4.1: build the unified clock (audit only; scoring loop refactor comes later)
    uni_clock = build_unified_clock(parquet_root, cfg["start"], cfg["end"], symbols)
    print(f"[Clock] unified minutes = {len(uni_clock)} from {uni_clock.min() if len(uni_clock) else '∅'} "
          f"to {uni_clock.max() if len(uni_clock) else '∅'}")

    # Optional: persist for debugging/inspection alongside portfolio outputs
    arte_root = _resolve_path(cfg.get("artifacts_root", "artifacts/a2"), create=True, is_dir=True)
    (arte_root / "portfolio").mkdir(parents=True, exist_ok=True)
    clock_out = arte_root / "portfolio" / "unified_clock.csv"
    if len(uni_clock):
        pd.Series(uni_clock, name="timestamp").to_csv(clock_out, index=False)
        print(f"[Clock] wrote unified clock → {clock_out}")
    else:
        print("[Clock] WARNING: unified clock is empty for the requested window/universe.")


    if not symbols:
        raise RuntimeError("No requested symbols were found under parquet/. Abort.")

    # Loop over symbols, run one-symbol workflow
    results = []
    for sym in symbols:
        sym_cfg = {**cfg, "symbol": sym}  # pass symbol to runner
        log.info(f"=== Running backtest for {sym} ===")
        res = await _run_one_symbol(sym, sym_cfg)
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



        # Phase 4.2: attach modeled costs (commission + half-spread + impact)
        # If your per-trade schema already contains 'half_spread_usd' or 'adv_frac',
        # pass them via half_spread_col / adv_pct_col below.
        trd_df = _apply_modeled_costs_to_trades(
            trd_df,
            cfg=cfg,
            price_col="entry_price",
            qty_col="qty" if "qty" in trd_df.columns else ("shares" if "shares" in trd_df.columns else "quantity"),
            half_spread_col="half_spread_usd" if "half_spread_usd" in trd_df.columns else None,
            adv_pct_col="adv_frac" if "adv_frac" in trd_df.columns else None,
        )

        # --- Phase 4.5: apply ledger & record portfolio columns on fills ------------
        # Initialize ledger using cfg equity; set gross/net limits relative to equity
        equity0 = float(cfg.get("equity", 100_000.0))
        ledger = PortfolioLedger(cash=equity0)



        limits = RiskLimits(
            max_gross=equity0 * float(cfg.get("max_gross_frac", 0.5)),
            max_net=equity0 * float(cfg.get("max_net_frac", 0.2)),
            per_symbol_cap=equity0 * float(cfg.get("per_symbol_cap_frac", 0.15)),
            sector_cap=equity0 * float(cfg.get("sector_cap_frac", 0.25)),
            max_concurrent=int(cfg.get("max_concurrent", 5)),
            daily_stop=-equity0 * float(cfg.get("daily_stop_frac", 0.02)),
            streak_stop=int(cfg.get("streak_stop", 0)),
        )
        risk = RiskEngine(limits)

        port_rows = []
        for i, row in trd_df.sort_values("entry_ts").iterrows():
            # entry leg
            entry = TradeFill(
                symbol=str(row["symbol"]),
                ts=pd.to_datetime(row["entry_ts"], utc=True),
                side=+1 if row.get("qty", 0.0) > 0 else -1,
                qty=abs(float(row.get("qty", 0.0))),
                price=float(row["entry_price"]),
                fees=float(row.get("modeled_cost_total", 0.0)) * 0.5,  # half on entry
            )
            ledger.on_fill(entry)
            # exit leg
            exitf = TradeFill(
                symbol=str(row["symbol"]),
                ts=pd.to_datetime(row["exit_ts"], utc=True),
                side=-entry.side,
                qty=entry.qty,
                price=float(row["exit_price"]),
                fees=float(row.get("modeled_cost_total", 0.0)) * 0.5,  # half on exit
            )
            ledger.on_fill(exitf)

            snap = ledger.snapshot_row()
            port_rows.append({**row.to_dict(), **snap})

        # augment trades with portfolio columns
        trd_df = pd.DataFrame(port_rows) if port_rows else trd_df

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

            # --- Phase 4.7: portfolio-level metrics (turnover, exposure, DD) -------
            def _compute_portfolio_metrics(trades_df: pd.DataFrame,
                                           equity_curve: pd.Series | None,
                                           equity0: float) -> dict:
                import numpy as _np
                import pandas as _pd

                if trades_df.empty:
                    return {
                        "n_trades": 0, "win_rate": 0.0, "turnover": 0.0,
                        "avg_concurrent_positions": 0.0, "max_drawdown": 0.0,
                    }

                df = trades_df.copy()
                # Coerce types
                for c in ("entry_ts", "exit_ts"):
                    if c in df.columns:
                        df[c] = _pd.to_datetime(df[c], utc=True, errors="coerce")
                for c in ("entry_price", "exit_price", "qty", "realized_pnl_after_costs", "realized_pnl"):
                    if c in df.columns:
                        df[c] = _pd.to_numeric(df[c], errors="coerce")

                # Round-trip dollars (simple turnover proxy)
                notional_in = (df["qty"].abs() * df["entry_price"]).fillna(0.0)
                notional_out = (df["qty"].abs() * df["exit_price"]).fillna(0.0)
                turnover_dollars = float((notional_in + notional_out).sum())
                turnover = (turnover_dollars / max(equity0, 1e-9))

                # Outcomes (prefer after-costs if present)
                pnl_col = "realized_pnl_after_costs" if "realized_pnl_after_costs" in df.columns else "realized_pnl"
                wins = (df[pnl_col] > 0).mean() if len(df) else 0.0

                # Concurrent exposure: average open positions across the run window
                # Build minute grid from entries/exits, count overlaps
                if df[["entry_ts", "exit_ts"]].notna().all().all():
                    # 1-minute grid between min(entry) and max(exit)
                    t0, t1 = df["entry_ts"].min(), df["exit_ts"].max()
                    grid = _pd.date_range(t0.floor("T"), t1.ceil("T"), freq="T", tz="UTC")
                    # fast interval counting
                    starts = _pd.Series(0, index=grid, dtype="int64")
                    ends = _pd.Series(0, index=grid, dtype="int64")
                    for _, r in df.iterrows():
                        es = r["entry_ts"].floor("T")
                        xs = r["exit_ts"].floor("T")
                        if es in starts.index: starts[es] += 1
                        if xs in ends.index:   ends[xs] += 1
                    open_ct = starts.cumsum() - ends.cumsum().shift(fill_value=0)
                    avg_conc = float(open_ct.mean())
                else:
                    avg_conc = 0.0

                # Max drawdown from equity curve (after realized PnL only)
                if equity_curve is not None and len(equity_curve) > 0:
                    eq = equity_curve.astype(float).copy()
                    roll_max = eq.cummax()
                    dd = (eq - roll_max)
                    max_dd = float(dd.min())  # negative number (depth)
                else:
                    max_dd = 0.0

                return {
                    "n_trades": int(len(df)),
                    "win_rate": float(wins),
                    "turnover": float(turnover),
                    "avg_concurrent_positions": float(avg_conc),
                    "max_drawdown": float(max_dd),
                }

            # Compute & write portfolio_metrics.json alongside equity curve
            metrics = _compute_portfolio_metrics(trd_df, curve if 'curve' in locals() else None, equity0)
            metrics_path = port_dir / "portfolio_metrics.json"
            metrics_path.write_text(json.dumps(metrics, indent=2))
            print(f"[Portfolio] metrics → {metrics_path} :: "
                  f"trades={metrics['n_trades']} win_rate={metrics['win_rate']:.2%} "
                  f"turnover={metrics['turnover']:.2f} avg_open={metrics['avg_concurrent_positions']:.2f} "
                  f"maxDD={metrics['max_drawdown']:.2f}")

        else:
            print("[Portfolio] skipped equity curve (missing exit_ts or realized_pnl)")

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
            # Phase 4.7 extra soft checks
            port_dir = _resolve_path(cfg.get("artifacts_root", "artifacts/a2"), create=True, is_dir=True) / "portfolio"
            metrics_json = port_dir / "portfolio_metrics.json"
            print("has portfolio_metrics.json:", metrics_json.exists())
            if (port_dir / "equity_curve.csv").exists():
                ec = pd.read_csv(port_dir / "equity_curve.csv", parse_dates=["exit_ts"])
                print("equity curve rows:", len(ec))

    else:
        print("[Phase-4] Note: could not find per-fold outputs to consolidate into decisions/trades. "
              "Backtest completed, but portfolio acceptance checks were skipped.")


    # ─── Step 4.8: evaluate readiness (requires portfolio/ files & costs ON) ───
    try:
        port_dir = _resolve_path(cfg.get("artifacts_root", "artifacts/a2"), create=True, is_dir=True) / "portfolio"
        promo = _promotion_checks_step4_8(port_dir=port_dir, cfg=cfg)
        print(f"[4.8] passed={promo['passed']}  "
              f"multi={promo['share_multisymbol']:.3f}  "
              f"slip_bps={promo['slippage_median_bps']:.2f}")
    except Exception as e:
        print(f"[4.8] Promotion checks skipped due to error: {e!r}")


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