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

from feature_engineering.utils.artifacts_root import resolve_artifacts_root
from pathlib import Path


import pyarrow.dataset as ds
from prediction_engine.prediction_engine.artifacts.manager import ArtifactManager
import json
from feature_engineering.utils.timegrid import grid_audit_to_json

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

from feature_engineering.utils.parquet_utc import write_parquet_utc


import logging
import sys
from pathlib import Path

from feature_engineering.utils.artifacts_root import resolve_artifacts_root


# at top of scripts/run_backtest.py
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.types as pat




from prediction_engine.run_context import RunContext



import builtins

_NOISE_PREFIXES = (
    "[Cal]",
    "[EV]",
    "[Cost]",
    "--- EXECUTING CORRECT BATCH_TOP_K METHOD ---",
)

def _install_print_filter(enable_verbose: bool) -> None:
    if enable_verbose:
        return
    _orig_print = builtins.print

    def _quiet_print(*args, **kwargs):
        if args and isinstance(args[0], str):
            s = args[0]
            if s.startswith(_NOISE_PREFIXES) or "EXECUTING CORRECT BATCH_TOP_K" in s:
                return
        return _orig_print(*args, **kwargs)

    builtins.print = _quiet_print




# --- Task3: run_context.json is the canonical root contract -------------------
import json
from datetime import datetime, timezone
from pathlib import Path

def _write_run_context_json(artifacts_root: Path, payload: dict) -> None:
    artifacts_root = Path(artifacts_root)
    artifacts_root.mkdir(parents=True, exist_ok=True)

    out_path = artifacts_root / "run_context.json"
    tmp_path = artifacts_root / "run_context.json.tmp"

    # Make payload stable + explicit
    payload = dict(payload)
    payload.setdefault("written_at_utc", datetime.now(timezone.utc).isoformat())
    payload["artifacts_root"] = str(artifacts_root)

    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(out_path)  # atomic on Windows if same volume




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
    #uni = pd.to_datetime(uni, utc=True).floor("T").sort_values().unique()
    uni = pd.to_datetime(uni, utc=True).floor("min").sort_values().unique()

    return pd.DatetimeIndex(uni, tz="UTC")


def _timestamp_overlap_share(
    bars: pd.DataFrame,
    *,
    ts_col: str = "timestamp",
    symbol_col: str = "symbol",
    min_symbols: int = 2,
) -> float:
    """
    Share of timestamps that have >= min_symbols symbols present.
    Example: min_symbols=2 => percentage of minutes where at least 2 symbols traded.
    """
    if bars is None or bars.empty:
        return 0.0
    if ts_col not in bars.columns or symbol_col not in bars.columns:
        return 0.0

    b = bars[[ts_col, symbol_col]].copy()
    b[ts_col] = pd.to_datetime(b[ts_col], utc=True, errors="coerce")
    b = b.dropna(subset=[ts_col, symbol_col])

    if b.empty:
        return 0.0

    # Count distinct symbols per timestamp
    counts = b.groupby(ts_col)[symbol_col].nunique()
    if len(counts) == 0:
        return 0.0

    good = (counts >= int(min_symbols)).sum()
    total = len(counts)
    return float(good) / float(total)


from typing import Any, Dict, List
from prediction_engine.testing_validation.walkforward import WalkForwardRunner


# ---- 5.3: callable backtest entrypoint (fold TEST runner) --------------------
from typing import Any, Dict

def run_batch(
    cfg: dict,
    *,
    artifacts_dir: str | Path | None = None,
    ev_artifacts_dir: str | Path | None = None,
) -> dict:
    """
    Canonical, fold-safe batch backtest runner.

    Guarantees:
      - single artifacts root
      - no duplicated resolution logic
      - run_context.json always written
      - out_dir always defined
    """

    # ------------------------------------------------------------------
    # Imports
    # ------------------------------------------------------------------
    from pathlib import Path
    from datetime import datetime, timezone
    import json
    import pandas as pd

    from feature_engineering.utils.artifacts_root import resolve_artifacts_root
    from prediction_engine.run_context import RunContext

    # ------------------------------------------------------------------
    # Phase 1.1 — Resolve ONE artifacts root
    # ------------------------------------------------------------------

    if artifacts_dir is not None:
        # Fold mode: caller provides leaf directory (e.g. fold_01)
        artifacts_root = Path(artifacts_dir).expanduser().resolve()
        artifacts_root.mkdir(parents=True, exist_ok=True)
        cfg["artifacts_root"] = str(artifacts_root)
        #print(f"[RunContext] (fold) artifacts_root={artifacts_root}")

    else:
        # Top-level mode: create run directory under canonical base
        base_root = resolve_artifacts_root(cfg, create=True)
        if not base_root.is_absolute():
            raise AssertionError(
                f"[Phase1.1] artifacts_root must be absolute, got: {base_root}"
            )

        run_id = str(cfg.get("run_id") or cfg.get("RUN_ID") or "").strip()
        if not run_id:
            raise RuntimeError("[Phase1.1] Missing run_id in cfg")

        run_ctx = RunContext.create(
            run_id=run_id,
            cfg={"artifacts_root": str(base_root)},
            universe_hash=str(cfg.get("universe_hash") or ""),
            window=f"{cfg.get('start')}→{cfg.get('end')}",
        )

        artifacts_root = run_ctx.run_dir
        cfg["artifacts_root"] = str(artifacts_root)

        print(f"[RunContext] artifacts_root(base)={base_root}")
        #print(f"[RunContext] run_dir={artifacts_root}")

        # Hard forbid scripts\artifacts for BOTH base_root and artifacts_root
        base_norm = str(base_root).lower().replace("/", "\\")
        run_norm = str(artifacts_root).lower().replace("/", "\\")
        if "\\scripts\\artifacts" in base_norm:
            raise AssertionError(f"[Phase1.1] Forbidden base_root: {base_root}")
        if "\\scripts\\artifacts" in run_norm:
            raise AssertionError(f"[Phase1.1] Forbidden run_dir: {artifacts_root}")

    # Hard guards
    artifacts_root = artifacts_root.resolve()
    if "\\scripts\\artifacts" in str(artifacts_root).lower().replace("/", "\\"):
        raise AssertionError(
            f"[Phase1.1] Forbidden artifacts location: {artifacts_root}"
        )

    # Unified output directory alias (kept for backward compatibility)
    out_dir = artifacts_root

    # ------------------------------------------------------------------
    # EV artifacts directory
    # ------------------------------------------------------------------
    if ev_artifacts_dir is not None:
        ev_dir = Path(ev_artifacts_dir).expanduser().resolve()
        ev_dir.mkdir(parents=True, exist_ok=True)
    else:
        ev_dir = artifacts_root / "ev"
        ev_dir.mkdir(parents=True, exist_ok=True)

    # Back-compat globals (some legacy helpers still expect these)
    globals()["_ARTIFACTS_ROOT"] = artifacts_root
    globals()["_EV_DIR"] = ev_dir

    # ------------------------------------------------------------------
    # Phase 1.1 observables (required by Task3)
    # ------------------------------------------------------------------
    (artifacts_root / "_ARTIFACTS_ROOT.txt").write_text(
        str(artifacts_root), encoding="utf-8"
    )

    run_context = {
        "run_id": cfg.get("run_id"),
        "artifacts_root": str(artifacts_root),
        "start": str(cfg.get("start")),
        "end": str(cfg.get("end")),
        "universe": cfg.get("universe"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (artifacts_root / "run_context.json").write_text(
        json.dumps(run_context, indent=2),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # Universe + bars loading
    # ------------------------------------------------------------------
    u = cfg.get("universe")
    if hasattr(u, "symbols"):
        symbols = list(u.symbols)
    elif isinstance(u, (list, tuple)):
        symbols = [str(s).upper() for s in u]
    else:
        raise TypeError("cfg['universe'] must be StaticUniverse or list[str]")

    start, end = cfg["start"], cfg["end"]
    pq_root = _resolve_path(cfg.get("parquet_root", "parquet"))

    print("[PARQUET] root:", pq_root)

    bars = []
    for s in symbols:
        df_s = _load_bars_for_symbol(
            {"parquet_root": str(pq_root), "start": start, "end": end}, s
        )
        if not df_s.empty:
            bars.append(df_s)

    bars = (
        pd.concat(bars, ignore_index=True)
        if bars
        else pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume", "symbol"]
        )
    )

    # Grid audit artifact (Task2)
    if hasattr(bars, "attrs") and "grid_audit" in bars.attrs:
        from feature_engineering.utils.timegrid import grid_audit_to_json

        audit = grid_audit_to_json(bars.attrs["grid_audit"])
        (artifacts_root / "grid_audit.json").write_text(
            json.dumps(audit, indent=2, default=str)
        )

    # ------------------------------------------------------------------
    # Empty-bar fast exit
    # ------------------------------------------------------------------
    if bars.empty:
        pd.DataFrame(
            columns=["timestamp", "symbol", "p_raw", "p_cal", "mu", "sigma", "target_qty"]
        ).to_parquet(out_dir / "decisions.parquet", index=False)

        pd.DataFrame(
            columns=[
                "symbol",
                "entry_ts",
                "entry_price",
                "exit_ts",
                "exit_price",
                "qty",
                "realized_pnl",
            ]
        ).to_parquet(out_dir / "trades.parquet", index=False)

        (out_dir / "signals_out.csv").write_text("")
        (out_dir / "equity.csv").write_text("timestamp,equity\n")
        (out_dir / "portfolio_metrics.json").write_text(
            json.dumps({"n_trades": 0}, indent=2)
        )

        return {"n_trades": 0, "symbols": symbols}

    # ------------------------------------------------------------------
    # Feature engineering (fold-fitted pipeline only)
    # ------------------------------------------------------------------
    feats_df = cfg.get("test_features_df")
    if feats_df is None:
        from feature_engineering.pipelines.core import CoreFeaturePipeline
        from feature_engineering.utils.timegrid import standardize_bars_to_grid

        bars = bars.copy()
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce")

        bars, _ = standardize_bars_to_grid(
            bars,
            symbol_col="symbol",
            ts_col="timestamp",
            freq="60s",
            expected_freq_s=60,
            fill_volume_zero=True,
            keep_ohlc_nan=True,
        )

        # ------------------------------------------------------------------
        # Phase 1.1 HARD GATE: multi-symbol timestamp overlap
        # Only enforce when this run is a multi-symbol portfolio run.
        # Walkforward fold runs are often single-symbol and must not trip this.
        # ------------------------------------------------------------------
        enforce_overlap_gate = bool(cfg.get("phase11_enforce_multisymbol_overlap_gate", False))

        n_syms = int(bars["symbol"].nunique()) if "symbol" in bars.columns else 0
        overlap = _timestamp_overlap_share(bars, min_symbols=2)

        # Always write audit (useful debugging), but only hard-fail when enforced.
        (artifacts_root / "overlap_audit.json").write_text(
            json.dumps(
                {
                    "enforced": enforce_overlap_gate,
                    "min_symbols": 2,
                    "overlap_share": overlap,
                    "min_required": float(cfg.get("min_overlap_share_ge2", 0.10)),
                    "n_rows": int(len(bars)),
                    "n_symbols": n_syms,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            f"[Gate] overlap_share(min_symbols>=2)={overlap:.4f} "
            f"(min_required={float(cfg.get('min_overlap_share_ge2', 0.10)):.2f}) "
            f"n_symbols={n_syms} enforced={enforce_overlap_gate}"
        )

        if enforce_overlap_gate:
            if n_syms < 2:
                raise RuntimeError(
                    f"[Phase1.1][HARD FAIL] Overlap gate enforced but run has n_symbols={n_syms}. "
                    f"This gate is only valid for multi-symbol portfolio runs."
                )

            min_overlap = float(cfg.get("min_overlap_share_ge2", 0.10))
            if overlap < min_overlap:
                raise RuntimeError(
                    f"[Phase1.1][HARD FAIL] Overlap gate failed: share(timestamps with >=2 symbols)={overlap:.4f} "
                    f"< {min_overlap:.2f}. Fix ingestion/grid alignment before producing portfolio artifacts."
                )

        pipe = CoreFeaturePipeline(
            parquet_root=Path(cfg.get("prepro_dir", ""))
        )
        feats_df = pipe.transform_mem(bars)

    # ------------------------------------------------------------------
    # EV scoring
    # ------------------------------------------------------------------
    from prediction_engine.ev_engine import EVEngine
    from prediction_engine.tx_cost import BasicCostModel

    ev = EVEngine.from_artifacts(
        ev_dir,
        cost_model=None if cfg.get("debug_no_costs") else BasicCostModel(),
    )

    pca_cols = [c for c in feats_df.columns if c.startswith("pca_")]
    if not pca_cols:
        raise RuntimeError("No PCA columns available for EV scoring")

    res = vectorize_minute_batch(ev, feats_df, pca_cols)
    decisions = getattr(res, "frame", res)

    # ------------------------------------------------------------------
    # Sizing + simulation
    # ------------------------------------------------------------------
    decisions = _apply_sizer_to_decisions(
        decisions, bars, cfg, target_qty_col="target_qty"
    )

    #trades = _simulate_trades_from_decisions(decisions, bars)
    # --- Execution rules (required) ---
    rules = cfg.get("execution_rules")

    if rules is None:
        raise RuntimeError(
            "[run_batch] Missing cfg['execution_rules'] required by "
            "_simulate_trades_from_decisions"
        )

    trades = _simulate_trades_from_decisions(
        decisions,
        bars,
        rules=rules,
    )

    trades = _apply_modeled_costs_to_trades(trades, cfg)

    # ------------------------------------------------------------------
    # Persist outputs
    # ------------------------------------------------------------------
    decisions.to_parquet(out_dir / "decisions.parquet", index=False)
    trades.to_parquet(out_dir / "trades.parquet", index=False)
    decisions.to_csv(out_dir / "signals_out.csv", index=False)

    # Equity
    if trades.empty:
        equity = pd.DataFrame(
            {"timestamp": [], "equity": []}, dtype="datetime64[ns, UTC]"
        )
    else:
        equity = trades[["exit_ts", "realized_pnl_after_costs"]].rename(
            columns={"exit_ts": "timestamp", "realized_pnl_after_costs": "equity"}
        )
        equity["timestamp"] = pd.to_datetime(equity["timestamp"], utc=True)
        equity["equity"] = equity["equity"].cumsum()

    equity.to_csv(out_dir / "equity.csv", index=False)

    metrics = {
        "symbols": symbols,
        "start": str(start),
        "end": str(end),
        "n_decisions": int(len(decisions)),
        "n_trades": int(len(trades)),
        "gross_pnl": float(
            trades.get("realized_pnl", pd.Series(dtype=float)).sum()
        ),
        "net_pnl": float(
            trades.get("realized_pnl_after_costs", pd.Series(dtype=float)).sum()
        ),
    }

    (out_dir / "portfolio_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    return metrics




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
from scripts.rebuild_artefacts import rebuild_if_needed, _dbg  # NEW
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
from prediction_engine.scoring.batch import vectorize_minute_batch, score_batch  # NEW (Step 4.6)


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


# --- timestamp normalization helper (tz-naive in UTC on disk) ---
'''def _coerce_ts_cols(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
    """
    Parse any present timestamp columns as UTC, then DROP the timezone so that
    Parquet stores tz-naive datetime64[ns] interpreted as UTC.
    This avoids 'Cannot mix tz-aware with tz-naive' errors later.
    """
    for c in cols:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce", utc=True)
            df[c] = s.dt.tz_localize(None)   # tz-naive, still in UTC
    return df
'''

# --- timestamp normalization helper (tz-aware UTC end-to-end) ---
'''def _coerce_ts_cols(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
    """
    Parse any present timestamp columns as tz-aware UTC (datetime64[ns, UTC]).
    DO NOT strip tz. Task 6 invariant: artifact boundaries must preserve UTC awareness.
    """
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    return df
'''

from feature_engineering.utils.time import ensure_utc_timestamp_col

class TimestampDtypeError(RuntimeError):
    pass

def _enforce_utc_ts_cols(df: pd.DataFrame, cols: tuple[str, ...], who: str) -> pd.DataFrame:
    """
    Enforce tz-aware UTC timestamps for any present timestamp columns.
    HARD FAIL if coercion creates NaT for any *required* timestamp column.
    """
    out = df.copy()
    for c in cols:
        if c in out.columns:
            try:
                ensure_utc_timestamp_col(out, c, who=who)
            except Exception as e:
                raise TimestampDtypeError(f"{who} failed UTC enforcement for col={c}: {e}") from e
    return out

def _drop_if_all_nat(df: pd.DataFrame, col: str, who: str) -> pd.DataFrame:
    """
    Some timestamp-like columns are optional (e.g., decision_ts). If present but
    completely invalid (all NaT after parse), drop it (and log) rather than killing the run.
    """
    if col not in df.columns:
        return df
    s = pd.to_datetime(df[col], errors="coerce", utc=True)
    if s.isna().all():
        print(f"[UTC] {who}: dropping optional ts col '{col}' because all values are NaT after coercion")
        return df.drop(columns=[col])
    # otherwise keep it and let the strict enforcer validate it
    return df




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
    dec_parts: list[pd.DataFrame] = []
    dec_patterns = [
        str(artifacts_root / "**" / "decisions.parquet"),
        str(artifacts_root / "**" / "decisions_*.parquet"),
        str(artifacts_root / "**" / "decisions.csv"),
        str(artifacts_root / "**" / "decisions_*.csv"),
        str(artifacts_root / "**" / "signals.parquet"),
        str(artifacts_root / "**" / "signals_*.parquet"),
        str(artifacts_root / "**" / "signals.csv"),
        str(artifacts_root / "**" / "signals_*.csv"),
    ]
    for pat in dec_patterns:
        for fp in glob.glob(pat, recursive=True):
            try:
                p_fp = Path(fp)

                # 1) Skip files sitting directly under artifacts_root (we only want per-fold/symbol parts)
                if p_fp.parent.resolve() == artifacts_root.resolve():
                    continue

                # Skip portfolio mirrors and already-consolidated outputs
                if "portfolio" in p_fp.parts:
                    continue
                if p_fp.name == "decisions.parquet" and p_fp.parent.resolve() == artifacts_root.resolve():
                    continue

                # 2) Skip truly tiny/placeholder parquet
                if p_fp.suffix == ".parquet" and p_fp.stat().st_size < 100:
                    continue

                if fp.endswith(".parquet"):
                    dec_parts.append(pd.read_parquet(fp))
                elif fp.endswith(".csv"):
                    dec_parts.append(pd.read_csv(fp))
            except Exception:
                # tolerate partial/corrupt fold outputs
                pass

    if dec_parts:
        '''dec = pd.concat(dec_parts, ignore_index=True).drop_duplicates()
        # normalize timestamps to tz-naive UTC on disk (single pass, no re-adding tz!)
        dec = _coerce_ts_cols(dec, ("timestamp", "decision_ts", "entry_ts", "exit_ts"))
        # light schema hygiene
        if "symbol" in dec.columns:
            dec["symbol"] = dec["symbol"].astype("string")
        #dec.to_parquet(decisions_out, index=False)

        # --- Task6: decisions timestamp schema repair (decision_ts optional but must be valid if present) ---
        # decision_ts is often missing/invalid in some paths. Define a deterministic policy:
        #   - if decision_ts exists but is mostly null/unparseable, fill from timestamp
        #   - if still invalid after fill, drop it (do not let optional cols break run)
        if "decision_ts" in dec.columns:
            # Try parsing; count how many become NaT
            parsed = pd.to_datetime(dec["decision_ts"], errors="coerce", utc=True)
            nat = int(parsed.isna().sum())

            # If any NaT, fill from canonical timestamp where possible
            if nat > 0 and "timestamp" in dec.columns:
                ts_fallback = pd.to_datetime(dec["timestamp"], errors="coerce", utc=True)
                parsed = parsed.fillna(ts_fallback)
                nat2 = int(parsed.isna().sum())

                if nat2 > 0:
                    # Still invalid after fallback → drop (optional column)
                    print(
                        f"[UTC][WARN] decision_ts invalid after fallback (NaT={nat2}); dropping column from consolidated decisions")
                    dec = dec.drop(columns=["decision_ts"])
                else:
                    dec["decision_ts"] = parsed
            else:
                dec["decision_ts"] = parsed
        # --- end Task6 decisions repair ---

        write_parquet_utc(
            dec,
            decisions_out,
            timestamp_cols=("timestamp", "decision_ts", "entry_ts", "exit_ts"),
        )'''

        dec = pd.concat(dec_parts, ignore_index=True).drop_duplicates()

        # Optional column cleanup BEFORE strict enforcement:
        dec = _drop_if_all_nat(dec, "decision_ts", who="[Consolidate decisions]")

        # Required columns must be valid tz-aware UTC:
        # - timestamp is REQUIRED for decisions artifacts
        if "timestamp" not in dec.columns:
            raise RuntimeError("[UTC] decisions missing required 'timestamp' column at consolidation")

        dec = _enforce_utc_ts_cols(dec, ("timestamp",), who="[Consolidate decisions REQUIRED]")
        #dec = _enforce_utc_ts_cols(dec, ("decision_ts",), who="[Consolidate decisions OPTIONAL]")
        print(f"[UTC][DEBUG] decisions columns={list(dec.columns)} rows={len(dec)}")
        if "decision_ts" in dec.columns:
            print(f"[UTC][DEBUG] decision_ts dtype before={dec['decision_ts'].dtype}")

        # decision_ts is optional: keep only if it can be fully coerced; otherwise drop
        if "decision_ts" in dec.columns:
            coerced = pd.to_datetime(dec["decision_ts"], errors="coerce", utc=True)

            # Optional fallback: if timestamp exists, fill missing decision_ts from timestamp
            if "timestamp" in dec.columns:
                coerced = coerced.fillna(pd.to_datetime(dec["timestamp"], errors="coerce", utc=True))

            if coerced.isna().any():
                nat = int(coerced.isna().sum())
                print(
                    f"[UTC] [Consolidate decisions OPTIONAL] dropping decision_ts (NaT={nat} after coercion/fallback)")
                dec = dec.drop(columns=["decision_ts"])
            else:
                dec["decision_ts"] = coerced

        # light schema hygiene
        if "symbol" in dec.columns:
            dec["symbol"] = dec["symbol"].astype("string")

        write_parquet_utc(
            dec,
            decisions_out,
            timestamp_cols=("timestamp", "decision_ts"),
        )
        #decisions_path = decisions_out

        decisions_path = decisions_out
    else:
        decisions_path = None


    # --- trades/fills/blotter: pick whatever exists and unify ---
    trade_parts: list[pd.DataFrame] = []
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
                p_fp = Path(fp)

                # 1) Skip files sitting directly under artifacts_root (portfolio root or top-level placeholders)
                if p_fp.parent.resolve() == artifacts_root.resolve():
                    continue

                # 2) Skip truly tiny/placeholder parquet
                if p_fp.suffix == ".parquet" and p_fp.stat().st_size < 100:
                    continue

                # Skip portfolio mirrors and already-consolidated outputs
                if "portfolio" in p_fp.parts:
                    continue
                if p_fp.name == "trades.parquet" and p_fp.parent.resolve() == artifacts_root.resolve():
                    continue

                if fp.endswith(".parquet"):
                    trade_parts.append(pd.read_parquet(fp))
                elif fp.endswith(".csv"):
                    trade_parts.append(pd.read_csv(fp))
            except Exception:
                pass

    if trade_parts:
        '''trades = pd.concat(trade_parts, ignore_index=True).drop_duplicates()
        trades = _coerce_ts_cols(trades, ("entry_ts", "exit_ts", "timestamp"))
        if "symbol" in trades.columns:
            trades["symbol"] = trades["symbol"].astype("string")
        #trades.to_parquet(trades_out, index=False)

        # --- Task6: decisions timestamp schema repair (decision_ts optional but must be valid if present) ---
        # decision_ts is often missing/invalid in some paths. Define a deterministic policy:
        #   - if decision_ts exists but is mostly null/unparseable, fill from timestamp
        #   - if still invalid after fill, drop it (do not let optional cols break run)
        if "decision_ts" in dec.columns:
            # Try parsing; count how many become NaT
            parsed = pd.to_datetime(dec["decision_ts"], errors="coerce", utc=True)
            nat = int(parsed.isna().sum())

            # If any NaT, fill from canonical timestamp where possible
            if nat > 0 and "timestamp" in dec.columns:
                ts_fallback = pd.to_datetime(dec["timestamp"], errors="coerce", utc=True)
                parsed = parsed.fillna(ts_fallback)
                nat2 = int(parsed.isna().sum())

                if nat2 > 0:
                    # Still invalid after fallback → drop (optional column)
                    print(
                        f"[UTC][WARN] decision_ts invalid after fallback (NaT={nat2}); dropping column from consolidated decisions")
                    dec = dec.drop(columns=["decision_ts"])
                else:
                    dec["decision_ts"] = parsed
            else:
                dec["decision_ts"] = parsed
        # --- end Task6 decisions repair ---

        write_parquet_utc(
            trades,
            trades_out,
            timestamp_cols=("timestamp", "entry_ts", "exit_ts"),
        )

        trades_path = trades_out'''

        trades = pd.concat(trade_parts, ignore_index=True).drop_duplicates()

        # If you have any optional timestamp-like columns in trades, drop-if-all-NaT them here.
        # (Most common case: none; entry_ts/exit_ts are required if present in your schema.)

        # Enforce required timestamps if present/expected.
        # Typical trades artifact expects entry_ts and exit_ts; timestamp may or may not exist depending on producer.
        trades = _enforce_utc_ts_cols(trades, ("entry_ts", "exit_ts"), who="[Consolidate trades REQUIRED]")
        trades = _enforce_utc_ts_cols(trades, ("timestamp",), who="[Consolidate trades OPTIONAL]")

        if "symbol" in trades.columns:
            trades["symbol"] = trades["symbol"].astype("string")

        write_parquet_utc(
            trades,
            trades_out,
            timestamp_cols=("entry_ts", "exit_ts", "timestamp"),
        )
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



def _phase4_consistency_gate(
    *,
    run_id: str | None,
    artifacts_root: Path,
    searched_dirs: list[Path],
    decisions_paths_found: int,
    trades_paths_found: int,
    consolidated_decisions_written: bool,
    consolidated_trades_written: bool,
    report_phase_decisions_count: int,
    report_phase_trades_count: int,
) -> None:
    """
    Phase 4: hard-fail if consolidation/reporting contradicts itself.

    Rules (strict):
      - If report says 0 decisions but consolidated decisions exist -> fail
      - If report says >0 decisions but consolidated missing -> fail
      - Same for trades
      - If we claim no candidate paths were found but consolidated written -> fail
    """
    contradictions: list[str] = []

    # report vs consolidated
    if report_phase_decisions_count == 0 and consolidated_decisions_written:
        contradictions.append("report_phase_decisions_count==0 but consolidated_decisions_written==True")
    if report_phase_decisions_count > 0 and not consolidated_decisions_written:
        contradictions.append("report_phase_decisions_count>0 but consolidated_decisions_written==False")

    if report_phase_trades_count == 0 and consolidated_trades_written:
        contradictions.append("report_phase_trades_count==0 but consolidated_trades_written==True")
    if report_phase_trades_count > 0 and not consolidated_trades_written:
        contradictions.append("report_phase_trades_count>0 but consolidated_trades_written==False")

    # candidate discovery vs consolidated write
    if decisions_paths_found == 0 and consolidated_decisions_written:
        contradictions.append("decisions_paths_found==0 but consolidated_decisions_written==True")
    if trades_paths_found == 0 and consolidated_trades_written:
        contradictions.append("trades_paths_found==0 but consolidated_trades_written==True")

    if contradictions:
        searched_str = "\n  - " + "\n  - ".join(str(p) for p in searched_dirs) if searched_dirs else " (none)"
        raise RuntimeError(
            "[Phase4ConsistencyGate] Contradictory run detected.\n"
            f"run_id={run_id}\n"
            f"artifacts_root={artifacts_root}\n"
            f"searched_dirs={searched_str}\n"
            f"decisions_paths_found={decisions_paths_found}\n"
            f"trades_paths_found={trades_paths_found}\n"
            f"consolidated_decisions_written={consolidated_decisions_written}\n"
            f"consolidated_trades_written={consolidated_trades_written}\n"
            f"report_phase_decisions_count={report_phase_decisions_count}\n"
            f"report_phase_trades_count={report_phase_trades_count}\n"
            "contradictions:\n  - " + "\n  - ".join(contradictions)
        )




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

    # If the dataset has no timestamp column (e.g., stub files), short-circuit empty.
    if "timestamp" not in dataset.schema.names:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "symbol"])

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

    print(f"[LOAD] {symbol} rows={len(df)} cols={list(df.columns)}")
    print(f"[LOAD] {symbol} ts dtype={df['timestamp'].dtype} tz_aware={getattr(df['timestamp'].dt, 'tz', None)}")
    print(f"[LOAD] {symbol} min/max={df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"[LOAD] {symbol} dup_ts_removed={(table.num_rows - len(df))}")
    print(f"[LOAD] {symbol} sample_ts={df['timestamp'].head(3).tolist()} ... {df['timestamp'].tail(3).tolist()}")

    sym_dir = parquet_root / f"symbol={symbol}"
    if sym_dir.exists():
        parts = list(sym_dir.glob("year=*/month=*/*.parquet"))
        print(f"[LOAD] {symbol} parquet parts={len(parts)} (sample={parts[:2]})")

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
    #view = df_features.loc[df_features["timestamp"].dt.floor("T") == t]
    view = df_features.loc[df_features["timestamp"].dt.floor("min") == t]

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
    # in your CONFIG dict / yaml
    "verbose_print": False,

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
    #"unit_test_force_constant_p": 0.60,
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

CONFIG.update({
    # Sizer/ramping gates
    "p_gate_quantile": 0.60,      # was 0.55
    "full_p_quantile": 0.72,      # was 0.65
    #"sizer_cost_lambda": 1.8,     # was 1.2 (penalize marginal p more)

    "sizer_strategy": "score",
    "max_gross_frac": 0.10,  # already defaulted in RiskCaps
    "max_net_frac": 0.05,
    "per_symbol_cap_frac": 0.06,
    "max_concurrent": 8,  # was 5 in limits; try 8 to avoid starving
    "adv_cap_pct": 0.10,  # reduce from 0.20 during bring-up
    "sizer_cost_lambda": 1.2,  # leave as is for now

})




#import datetime as dt
#CONFIG["artifacts_root"] = f"artifacts/a2_{dt.datetime.utcnow():%Y%m%d_%H%M%S}"

# Phase 1.1: Do NOT set artifacts_root here.
# RunContext will choose a single canonical root once per run.
CONFIG.pop("artifacts_root", None)

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
    print(f"[AggFind] {sym} root={root} exists={root.exists()}")
    print(f"[AggFind] {sym} decisions candidates:", cand_dec[:3])
    print(f"[AggFind] {sym} trades candidates:", cand_trd[:3])

    return out



def _overlap_metrics(bars: pd.DataFrame, *, ts_col="timestamp", symbol_col="symbol") -> dict:
    """
    Returns:
      - global_share_ge2: share of timestamps that have >=2 symbols present
      - conditional_share_sparse_in_dense: of sparse-symbol timestamps, what fraction are also present in the densest symbol
      - per_symbol_minutes: dict(symbol -> count unique timestamps)
    """
    if bars is None or bars.empty:
        return {
            "global_share_ge2": 0.0,
            "conditional_share_sparse_in_dense": 0.0,
            "per_symbol_minutes": {},
        }

    df = bars[[ts_col, symbol_col]].copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col, symbol_col])
    if df.empty:
        return {
            "global_share_ge2": 0.0,
            "conditional_share_sparse_in_dense": 0.0,
            "per_symbol_minutes": {},
        }

    # unique minutes per symbol
    per_symbol = (
        df.drop_duplicates([symbol_col, ts_col])
          .groupby(symbol_col)[ts_col]
          .size()
          .to_dict()
    )

    # global share with >=2 symbols present
    counts = df.drop_duplicates([symbol_col, ts_col]).groupby(ts_col)[symbol_col].nunique()
    global_share_ge2 = float((counts >= 2).mean()) if len(counts) else 0.0

    # conditional overlap: sparse symbol timestamps in densest symbol timestamps
    if len(per_symbol) < 2:
        conditional = 0.0
    else:
        sparse_sym = min(per_symbol, key=per_symbol.get)
        dense_sym = max(per_symbol, key=per_symbol.get)

        sparse_ts = set(df.loc[df[symbol_col] == sparse_sym, ts_col].unique())
        dense_ts  = set(df.loc[df[symbol_col] == dense_sym, ts_col].unique())

        conditional = float(len(sparse_ts & dense_ts) / len(sparse_ts)) if sparse_ts else 0.0

    return {
        "global_share_ge2": global_share_ge2,
        "conditional_share_sparse_in_dense": conditional,
        "per_symbol_minutes": {str(k): int(v) for k, v in per_symbol.items()},
    }



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
# --- Phase 4.2: attach modeled costs to trades (commission + spread + impact)
def _apply_modeled_costs_to_trades(
    trades: pd.DataFrame,
    cfg: dict,
    *,
    price_col: str = "entry_price",
    exit_price_col: str = "exit_price",
    qty_col: str = "qty",
    half_spread_col: str | None = "half_spread_usd",
    adv_pct_col: str | None = "adv_frac",
) -> pd.DataFrame:
    """
    Add modeled costs and net PnL using flexible column names.
    Produces:
      - 'modeled_cost_total'
      - 'realized_pnl_after_costs'
    """
    # Defensive: allow None or empty
    if trades is None:
        return trades
    if len(trades) == 0:
        # ensure expected columns exist even when empty (nice for downstream)
        df = trades.copy()
        df["modeled_cost_total"] = 0.0
        df["realized_pnl_after_costs"] = 0.0
        return df

    df = trades.copy()

    def _col_as_series(col: str, default: float = 0.0) -> pd.Series:
        """Return df[col] coerced to numeric Series; if missing, a default Series."""
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            return s.fillna(default)
        return pd.Series(default, index=df.index, dtype=float)

    # Costless short-circuit
    if bool(cfg.get("debug_no_costs", False)):
        df["modeled_cost_total"] = 0.0
        if "realized_pnl" in df.columns:
            df["realized_pnl_after_costs"] = pd.to_numeric(df["realized_pnl"], errors="coerce").fillna(0.0)
        else:
            df["realized_pnl_after_costs"] = 0.0
        return df

    # Knobs
    spread_bp = float(cfg.get("spread_bp", 1.0))                      # half-spread (bps of price)
    commission_per_share = float(cfg.get("commission", 0.0))          # $/share
    slippage_bp = float(cfg.get("slippage_bp", 0.0))                  # bps
    impact_bps_per_frac = float(cfg.get("impact_bps_per_adv_frac", 25.0))  # bps per 1.0 ADV frac

    # Core columns (always Series)
    qty_raw = _col_as_series(qty_col, default=0.0)
    qty = qty_raw.abs()
    entry = _col_as_series(price_col, default=0.0)
    exit_ = _col_as_series(exit_price_col, default=0.0)
    mid = (entry + exit_) / 2.0

    # Half-spread $/share
    if half_spread_col and half_spread_col in df.columns:
        half_spread_usd = pd.to_numeric(df[half_spread_col], errors="coerce")
    else:
        half_spread_usd = pd.Series(np.nan, index=df.index, dtype=float)

    fallback_half_spread = (spread_bp / 1e4) * mid
    half_spread_usd = half_spread_usd.where(~half_spread_usd.isna(), fallback_half_spread).fillna(0.0)

    # ADV fraction (for impact)
    if adv_pct_col and adv_pct_col in df.columns:
        adv_frac = pd.to_numeric(df[adv_pct_col], errors="coerce").fillna(0.0)
    else:
        adv_frac = pd.Series(0.0, index=df.index, dtype=float)

    # Components (all positive)
    spread_cost = 2.0 * half_spread_usd * qty
    commission_cost = 2.0 * commission_per_share * qty
    slippage_cost = (slippage_bp / 1e4) * mid * 2.0 * qty
    impact_cost = (impact_bps_per_frac / 1e4) * mid * adv_frac * qty

    total = (spread_cost + commission_cost + slippage_cost + impact_cost).fillna(0.0)
    df["modeled_cost_total"] = total

    # Prefer realized_pnl if present; else compute naive RT PnL
    if "realized_pnl" in df.columns:
        base = pd.to_numeric(df["realized_pnl"], errors="coerce").fillna(0.0)
    else:
        # side derived from qty sign (if your qty stores signed direction)
        side = np.sign(qty_raw).replace(0.0, 1.0)  # treat 0 as +1 to avoid NaNs
        base = (exit_ - entry) * (side * qty)      # crude

    df["realized_pnl_after_costs"] = base - total
    return df


# Phase 4.3 – simulator & quotes
from prediction_engine.portfolio.order_sim import simulate_entry, simulate_exit, QuoteStats
from prediction_engine.portfolio.quotes import estimate_quote_stats_from_rolling



# put this with your other small helpers (e.g., above _simulate_trades_from_decisions)

def _ledger_snapshot_row(ledger):
    """
    Compatibility shim for ledgers without .snapshot_row().

    Tries, in order:
      1) ledger.snapshot_row() if present
      2) Collects common portfolio fields from attributes/callables:
         cash, equity, gross, net, day_pnl, open_pnl, closed_pnl, buying_power
         and a lightweight positions summary.
    Returns a dict suitable for merging into a trades row.
    """
    # Prefer native API if available
    if hasattr(ledger, "snapshot_row") and callable(getattr(ledger, "snapshot_row")):
        try:
            snap = ledger.snapshot_row()
            if isinstance(snap, dict):
                return snap
        except Exception:
            pass  # fall through to manual construction

    def _maybe(name, default=0.0):
        val = getattr(ledger, name, None)
        if callable(val):
            try:
                val = val()
            except Exception:
                val = None
        if val is None:
            return default
        try:
            return float(val)
        except Exception:
            return default

    # Positions: try mapping of symbols → {qty, avg_price} or a simple dict
    pos = getattr(ledger, "positions", None)
    if callable(pos):
        try:
            pos = pos()
        except Exception:
            pos = None
    if pos is None:
        pos = {}
    try:
        open_positions = int(sum(1 for p in pos.values()
                                 if getattr(p, "qty", 0) != 0 or (isinstance(p, dict) and p.get("qty", 0) != 0)))
    except Exception:
        open_positions = 0

    snap = {
        "cash":          _maybe("cash", 0.0),
        "equity":        _maybe("equity", 0.0),
        "gross":         _maybe("gross", 0.0),
        "net":           _maybe("net", 0.0),
        "day_pnl":       _maybe("day_pnl", 0.0),
        "open_pnl":      _maybe("open_pnl", 0.0),
        "closed_pnl":    _maybe("closed_pnl", 0.0),
        "buying_power":  _maybe("buying_power", 0.0),
        "open_positions": open_positions,
    }
    return snap


def _ledger_fill(ledger, fill):
    """
    Compatibility shim:
      - If ledger supports object-style (on_fill/record_fill), pass TradeFill as-is.
      - If ledger only has arg-style apply_fill(symbol, ts, qty, price, ...), decompose TradeFill.
      - As a last resort, try record(...) with kwargs.
    """
    # 1) Object-style APIs
    if hasattr(ledger, "on_fill"):
        return ledger.on_fill(fill)
    if hasattr(ledger, "record_fill"):
        return ledger.record_fill(fill)

    # 2) Arg-style API: apply_fill(symbol, ts, qty, price, side=?, fees=?)
    if hasattr(ledger, "apply_fill"):
        sym   = getattr(fill, "symbol", None)
        ts    = getattr(fill, "ts", None)
        qty   = float(getattr(fill, "qty", 0.0))
        price = float(getattr(fill, "price", 0.0))
        side  = int(getattr(fill, "side", 1))     # +1 long, -1 short
        fees  = float(getattr(fill, "fees", 0.0))

        # Try kwargs first (most descriptive)
        try:
            return ledger.apply_fill(symbol=sym, ts=ts, qty=qty, price=price, side=side, fees=fees)
        except TypeError:
            pass
        # Try positional without optional args
        try:
            return ledger.apply_fill(sym, ts, qty, price)
        except TypeError:
            pass
        # Try positional with side/fees appended
        try:
            return ledger.apply_fill(sym, ts, qty, price, side, fees)
        except TypeError:
            pass

    # 3) Very generic fallback: record(...)
    if hasattr(ledger, "record"):
        try:
            return ledger.record(fill)
        except TypeError:
            sym   = getattr(fill, "symbol", None)
            ts    = getattr(fill, "ts", None)
            qty   = float(getattr(fill, "qty", 0.0))
            price = float(getattr(fill, "price", 0.0))
            side  = int(getattr(fill, "side", 1))
            fees  = float(getattr(fill, "fees", 0.0))
            return ledger.record(symbol=sym, ts=ts, qty=qty, price=price, side=side, fees=fees)

    raise AttributeError(
        "PortfolioLedger has no usable fill method (tried on_fill, record_fill, apply_fill, record)."
    )



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
            "entry_open": px_e,
            "entry_price": ent.entry_price,
            "exit_ts": ex.exit_ts,
            "exit_open": px_x,
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
        base = _ledger_snapshot_row(ledger)
        # Derive positions view defensively
        positions_attr = getattr(ledger, "positions", {})
        try:
            pos_syms = [s for s, p in positions_attr.items()
                        if (getattr(p, "qty", 0) != 0) or (isinstance(p, dict) and p.get("qty", 0) != 0)]
            sym_notional = {
                s: abs((getattr(p, "qty", p.get("qty", 0))) * (getattr(p, "avg_price", p.get("avg_price", 0.0))))
                for s, p in positions_attr.items()
            }
        except Exception:
            pos_syms, sym_notional = [], {}

        return {
            **base,
            "positions": pos_syms,
            "symbol_notional": sym_notional,
            "sector_gross": {},  # local working view for this timestamp
            "open_positions": base.get("open_positions", len(pos_syms)),
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
    _install_print_filter(enable_verbose=bool(cfg.get("verbose_print", False)))
    if not bool(cfg.get("verbose_print", False)):
        logging.getLogger().setLevel(logging.WARNING)

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

    #mask = await detector(df_raw)

    mask = detector(df_raw)

    print(f"[Scanner] {sym} kept={int(mask.sum())}/{len(mask)}")

    # quick drift / sanity: compare basic OHLCV stats before vs after
    def _stats(tag, d):
        return {
            "n": len(d),
            "open_mean": float(d["open"].mean()),
            "open_std": float(d["open"].std(ddof=1) or 0),
            "vol_mean": float(d["volume"].mean()),
            "vol_std": float(d["volume"].std(ddof=1) or 0),
        }


    mask = pd.Series(mask, index=df_raw.index).astype(bool)

    pass_rate = mask.mean() * 100
    print(f"[Scanner KPI] {sym}: Bars passing filters: {mask.sum()} / {len(mask)} = {pass_rate:.2f}%")

    df_full = df_raw.copy()

    df_full = df_full.sort_values("timestamp").reset_index(drop=True)
    df_full["_bar_id"] = np.arange(len(df_full), dtype=np.int64)

    print("[Scanner] before:", _stats("before", df_full))
    print("[Scanner] after :", _stats("after", df_full.loc[mask]))

    from pathlib import Path
    parquet_root = _resolve_path(cfg.get("parquet_root", "parquet"))


    root = Path(parquet_root)

    # adjust pattern to your folder structure
    hits = list(root.rglob(f"*symbol={sym}*"))
    print(f"[FILES] requested={sym} hits={len(hits)}")
    for p in hits[:10]:
        print("   ", p)

    print(f"[SYMBOL-CHECK] requested={sym}")
    if "symbol" in df_full.columns:
        print("[SYMBOL-CHECK] df_full unique:", df_full["symbol"].unique()[:10], "nunique=",
              df_full["symbol"].nunique())
        print("[SYMBOL-CHECK] head symbols:", df_full["symbol"].head(5).tolist())
    else:
        print("[SYMBOL-CHECK] df_full has NO symbol column")

    print(f"[DATA-CHECK] {sym} df_full rows={len(df_full)} "
          f"min={df_full['timestamp'].min()} max={df_full['timestamp'].max()}")
    #print(f"[PARQUET-ROOT] {parquet_root}")
    print(f"[SYMBOL] {sym}")
    print(f"[DF] rows={len(df_full)} min={df_full['timestamp'].min()} max={df_full['timestamp'].max()}")
    print(f"[DF] unique_dates={df_full['timestamp'].dt.date.nunique()}")


    print("TEST: Are BBY and RRC actually the same data mistakenly labeled?")
    print(df_full[['timestamp', 'open', 'high', 'low', 'close']].head(3))
    print(df_full[['timestamp', 'open', 'high', 'low', 'close']].tail(3))

    from feature_engineering.utils.timegrid import standardize_bars_to_grid

    df_full, grid_audits = standardize_bars_to_grid(
        df_full,
        symbol_col="symbol",
        ts_col="timestamp",
        freq="60s",
        expected_freq_s=60,
        fill_volume_zero=True,
        keep_ohlc_nan=True,
        hard_fail_on_duplicates=False,
    )

    # optional: print audit summary (nice when debugging)
    if grid_audits:
        bad = [a for a in grid_audits if (a.median_delta_s_out not in (None, 60.0))]
        if bad:
            print("[GridAudit] non-60s after standardize:", bad)

    #df_raw = df_raw.loc[mask].reset_index(drop=True)

    df_raw = df_raw.loc[mask].copy()  # keep original index so fold merge can key on it

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

        # AFTER creating `dec` with p_raw/p_cal and before sizing:
        # Attach quick y for sign check using open->open over H bars
        H = int(cfg.get("horizon_bars", 3))
        tmp = df_full[["timestamp", "symbol", "open"]].copy()
        tmp = tmp.sort_values("timestamp").reset_index(drop=True)
        tmp["open_fwd"] = tmp["open"].shift(-H)
        tmp["y"] = (np.log(tmp["open_fwd"] / tmp["open"]) > 0.0).astype(float)
        dec = dec.merge(tmp[["timestamp", "symbol", "y"]], on=["timestamp", "symbol"], how="left")

        if bool(cfg.get("sign_check", False)):
            dec, flipped = _maybe_flip_probs(dec, p_col="p_cal", y_col="y")
            if flipped:
                print("[SignCheck] p_cal flipped (AUC(1-p) > AUC(p)) in unit-test path.")

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
        #artifacts_root=_resolve_path(cfg.get("artifacts_root", "artifacts/a2"), create=True, is_dir=True),
        artifacts_root=resolve_artifacts_root(cfg, create=True),
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

    print("[RAW DF] rows=", len(df_full))
    print("[RAW DF] ts dtype=", df_full["timestamp"].dtype)
    ts = pd.to_datetime(df_full["timestamp"], utc=True, errors="coerce")
    print("[RAW DF] min/max=", ts.min(), "→", ts.max())
    print("[RAW DF] NaT count=", ts.isna().sum())

    '''runner = WalkForwardRunner(
        artifacts_root=_resolve_path(cfg.get("artifacts_root", "artifacts/a2")),
        parquet_root=resolved_parquet_root,
        ev_artifacts_root=_resolve_path(cfg["artefacts"]),
        symbol=sym,
        horizon_bars=int(cfg.get("horizon_bars", 20)),
        longest_lookback_bars=int(cfg.get("longest_lookback_bars", 60)),
        p_gate_q=float(cfg.get("p_gate_quantile", 0.65)),
        full_p_q=float(cfg.get("full_p_quantile", 0.80)),
        debug_no_costs=bool(cfg.get("debug_no_costs", False)),
    )'''

    #from feature_engineering.utils.artifacts_root import resolve_artifacts_root

    #base_artifacts_root = resolve_artifacts_root(cfg, create=True) / "a2"

    # Canonical Phase 1.1 run directory: <repo_root>/artifacts/a2_<RUN_ID>
    base_artifacts_root = resolve_artifacts_root(cfg, run_id=RUN_ID, create=True)
    cfg["artifacts_root"] = str(base_artifacts_root)

    runner = WalkForwardRunner(
        artifacts_root=base_artifacts_root,
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

from sklearn.metrics import roc_auc_score

def _maybe_flip_probs(df: pd.DataFrame, p_col: str = "p_cal", y_col: str = "y", *, mark_col: str = "_p_flipped") -> tuple[pd.DataFrame, bool]:
    """
    If AUC(1-p) > AUC(p), flip probabilities in-place.
    Returns (df, flipped_bool). No-op if y missing or degenerate.
    """
    if p_col not in df.columns or y_col not in df.columns:
        df[mark_col] = False
        return df, False
    d = df.dropna(subset=[p_col, y_col])
    if len(d) < 50 or d[y_col].nunique() < 2:
        df[mark_col] = False
        return df, False
    try:
        auc = roc_auc_score(d[y_col], d[p_col])
        auc_inv = roc_auc_score(d[y_col], 1.0 - d[p_col])
        flipped = bool(auc_inv > auc)
        if flipped:
            df[p_col] = 1.0 - df[p_col]
        df[mark_col] = flipped
        return df, flipped
    except Exception:
        df[mark_col] = False
        return df, False



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
    #logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout)
    #log = logging.getLogger("backtest")

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout)
    log = logging.getLogger("backtest")
    _install_print_filter(enable_verbose=bool(cfg.get("verbose_print", False)))
    if not bool(cfg.get("verbose_print", False)):
        # Clamp root + likely chatty modules
        logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger("prediction_engine.scoring.batch").setLevel(logging.ERROR)

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


    # --- Task3: lock canonical artifacts_root for the entire run ---
    cfg["artifacts_root"] = str(arte_root)  # canonical absolute (repo-root based)

    print(f"[RunContext] artifacts_root={arte_root}")

    _write_run_context_json(
        arte_root,
        payload={
            "run_id": RUN_ID,
            "universe_hash": uhash,
            "universe_source": usrc,
            "window_start": str(cfg["start"]),
            "window_end": str(cfg["end"]),
            # optional extra provenance if you have it:
            "parquet_root": str(_resolve_path(cfg.get("parquet_root", "parquet"))),
        },
    )

    print(f"[Run] universe_hash={uhash} | source={usrc} | window={win_s}→{win_e} | artifacts={arte_root}")

    print("[RUN] mode=", BACKTEST_MODE, "run_id=", RUN_ID, "git=", GIT_SHA)
    print("[RUN] window=", cfg["start"], "→", cfg["end"], "horizon_bars=", cfg.get("horizon_bars"))
    print("[RUN] costs: commission=", cfg.get("commission"), "spread_bp=", cfg.get("spread_bp"), "slippage_bp=",
          cfg.get("slippage_bp"),
          "debug_no_costs=", cfg.get("debug_no_costs"))
    print("[RUN] gates: p_gate=", cfg.get("p_gate_quantile"), "p_full=", cfg.get("full_p_quantile"),
          "sizer_cost_lambda=", cfg.get("sizer_cost_lambda"), "strategy=", cfg.get("sizer_strategy"))

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

    # after uni_clock built
    cover = {}
    for s in symbols:
        df_s = _load_bars_for_symbol({"parquet_root": str(parquet_root), "start": cfg["start"], "end": cfg["end"]}, s)
        if df_s.empty:
            cover[s] = 0
        else:
            cover[s] = int(df_s["timestamp"].dt.floor("min").nunique())
    print("[Clock] per-symbol minute coverage:", cover)

    # Optional: persist for debugging/inspection alongside portfolio outputs
    '''arte_root = _resolve_path(cfg.get("artifacts_root", "artifacts/a2"), create=True, is_dir=True)
    (arte_root / "portfolio").mkdir(parents=True, exist_ok=True)
    clock_out = arte_root / "portfolio" / "unified_clock.csv"
    if len(uni_clock):
        pd.Series(uni_clock, name="timestamp").to_csv(clock_out, index=False)
        print(f"[Clock] wrote unified clock → {clock_out}")
    else:
        print("[Clock] WARNING: unified clock is empty for the requested window/universe.")
    '''


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
    #arte_root = _resolve_path(cfg.get("artifacts_root", "artifacts/a2"), create=True, is_dir=True)
    arte_root = Path(cfg["artifacts_root"])

    dec_df, trd_df = _aggregate_universe_outputs(arte_root, symbols)

    print("[Agg] decisions rows=", len(dec_df), "trades rows=", len(trd_df))
    if not dec_df.empty and "symbol" in dec_df.columns:
        print("[Agg] decisions by symbol:", dec_df["symbol"].value_counts().to_dict())
    if not trd_df.empty and "symbol" in trd_df.columns:
        print("[Agg] trades by symbol:", trd_df["symbol"].value_counts().to_dict())

    # Make a portfolio folder
    #port_dir = arte_root / "portfolio"
    #port_dir.mkdir(parents=True, exist_ok=True)

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
        '''trd_df = _apply_modeled_costs_to_trades(
            trd_df,
            cfg=cfg,
            price_col="entry_price",
            qty_col="qty" if "qty" in trd_df.columns else ("shares" if "shares" in trd_df.columns else "quantity"),
            half_spread_col="half_spread_usd" if "half_spread_usd" in trd_df.columns else None,
            adv_pct_col="adv_frac" if "adv_frac" in trd_df.columns else None,
        )'''

        trd_df = _apply_modeled_costs_to_trades(
            trd_df,
            cfg=cfg,
            price_col="entry_price",
            exit_price_col="exit_price",
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
            #ledger.on_fill(entry)

            _ledger_fill(ledger, entry)

            # exit leg
            exitf = TradeFill(
                symbol=str(row["symbol"]),
                ts=pd.to_datetime(row["exit_ts"], utc=True),
                side=-entry.side,
                qty=entry.qty,
                price=float(row["exit_price"]),
                fees=float(row.get("modeled_cost_total", 0.0)) * 0.5,  # half on exit
            )
            #ledger.on_fill(exitf)

            _ledger_fill(ledger, exitf)

            #snap = ledger.snapshot_row()
            #port_rows.append({**row.to_dict(), **snap})
            snap = _ledger_snapshot_row(ledger)
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
                    #grid = _pd.date_range(t0.floor("T"), t1.ceil("T"), freq="T", tz="UTC")
                    grid = _pd.date_range(t0.floor("min"), t1.ceil("min"), freq="min", tz="UTC")

                    # fast interval counting
                    starts = _pd.Series(0, index=grid, dtype="int64")
                    ends = _pd.Series(0, index=grid, dtype="int64")
                    for _, r in df.iterrows():
                        #es = r["entry_ts"].floor("T")
                        #xs = r["exit_ts"].floor("T")
                        es = r["entry_ts"].floor("min")
                        xs = r["exit_ts"].floor("min")
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

            def _perf_from_equity(equity_curve: pd.Series, equity0: float) -> dict:
                eq = equity_curve.astype(float).copy()
                if len(eq) < 2:
                    return {"CAGR": 0.0, "Sharpe": 0.0, "Sortino": 0.0, "Calmar": 0.0, "max_dd_abs": 0.0}
                eq_level = equity0 + eq
                t0, t1 = eq_level.index[0], eq_level.index[-1]
                years = max((t1 - t0).days / 365.25, 1e-9)
                ret_total = float(eq_level.iloc[-1] / equity0 - 1.0)
                CAGR = (1.0 + ret_total) ** (1.0 / years) - 1.0

                rets = eq_level.diff().fillna(0.0) / equity0
                mu = float(rets.mean());
                sigma = float(rets.std(ddof=1) or 1e-12)
                Sharpe = (mu / sigma) * (np.sqrt(len(rets)) / max(years ** 0.5, 1e-9))
                downside = rets[rets < 0.0]
                dsig = float(downside.std(ddof=1) or 1e-12)
                Sortino = (mu / dsig) * (np.sqrt(len(rets)) / max(years ** 0.5, 1e-9))

                roll_max = eq_level.cummax()
                dd = (eq_level - roll_max)
                max_dd_abs = float(abs(dd.min()))
                Calmar = (CAGR / (max_dd_abs / equity0)) if max_dd_abs > 0 else 0.0
                return {"CAGR": float(CAGR), "Sharpe": float(Sharpe), "Sortino": float(Sortino),
                        "Calmar": float(Calmar), "max_dd_abs": max_dd_abs}

            # Merge perf metrics if we have a curve
            try:
                equity0 = float(cfg.get("equity", 100_000.0))
                perf_extra = _perf_from_equity(curve if 'curve' in locals() else pd.Series(dtype=float), equity0)
                metrics = {**metrics, **perf_extra}
            except Exception as _e:
                print("[Portfolio] perf metrics skipped:", repr(_e))

            metrics_path = port_dir / "portfolio_metrics.json"
            metrics_path.write_text(json.dumps(metrics, indent=2))
            print(f"[Portfolio] metrics → {metrics_path} :: "
                  f"trades={metrics['n_trades']} win_rate={metrics['win_rate']:.2%} "
                  f"turnover={metrics['turnover']:.2f} avg_open={metrics['avg_concurrent_positions']:.2f} "
                  f"maxDD={metrics['max_drawdown']:.2f}")

        else:
            print("[Portfolio] skipped equity curve (missing exit_ts or realized_pnl)")




    # --- Phase-4: consolidate per-fold/per-symbol outputs into single files ---
    '''artifacts_root = _resolve_path(cfg.get("artifacts_root", "artifacts/a2"), create=True, is_dir=True)
    dec_path, trd_path = _consolidate_phase4_outputs(artifacts_root)

    # --- Always mirror consolidated outputs into portfolio/ for 4.8 checks ---
    port_dir = artifacts_root / "portfolio"'''

    # --- Phase-4: consolidate per-fold/per-symbol outputs into single files ---
    # Never re-resolve here. Use the run directory chosen at start.
    artifacts_root = Path(cfg["artifacts_root"]).resolve()

    # Hard gates (repeat here because Phase-4 is where the split used to occur)
    if not artifacts_root.is_absolute():
        raise RuntimeError(f"[Phase-4] artifacts_root must be absolute, got: {artifacts_root}")

    s = str(artifacts_root).lower().replace("/", "\\")
    if "\\scripts\\artifacts\\" in s:
        raise RuntimeError(f"[Phase-4] Illegal artifacts_root (scripts/artifacts): {artifacts_root}")

    rc = artifacts_root / "run_context.json"
    if not rc.exists():
        raise RuntimeError(f"[Phase-4] Refusing to consolidate: missing {rc}")

    dec_path, trd_path = _consolidate_phase4_outputs(artifacts_root)

    # Mirror consolidated outputs into portfolio/ under THIS SAME artifacts_root
    port_dir = artifacts_root / "portfolio"

    port_dir.mkdir(parents=True, exist_ok=True)

    if dec_path is not None:
        dec = pd.read_parquet(dec_path)
        dec.to_parquet(port_dir / "decisions.parquet", index=False)

    if trd_path is not None:
        trd = pd.read_parquet(trd_path)
        trd.to_parquet(port_dir / "trades.parquet", index=False)

    print(f"[Portfolio] mirrored consolidated → {port_dir}")

    # --- Ensure portfolio equity curve exists (even if earlier branch skipped) ---
    try:
        trd_p = port_dir / "trades.parquet"
        if trd_p.exists():
            trd_df = pd.read_parquet(trd_p)
            if not trd_df.empty and {"exit_ts", "realized_pnl"}.issubset(trd_df.columns):
                curve = equity_curve_from_trades(trd_df)
                (port_dir / "equity_curve.csv").parent.mkdir(parents=True, exist_ok=True)
                curve.to_csv(port_dir / "equity_curve.csv", header=["equity"])
                # also emit a simple equity.csv for legacy readers
                curve.rename_axis("timestamp").to_frame("equity").reset_index().to_csv(
                    port_dir / "equity.csv", index=False
                )

    except Exception as _e:
        print("[Portfolio] equity curve rebuild skipped:", repr(_e))

    # Soft acceptance checks: only probe if the files exist
    if dec_path is not None and trd_path is not None:
        dec = pd.read_parquet(dec_path)
        trd = pd.read_parquet(trd_path)

        print("decisions rows:", len(dec), "trades rows:", len(trd))
        print("symbols in decisions:", sorted(dec['symbol'].dropna().astype(str).unique().tolist()) if 'symbol' in dec.columns else "(no symbol column)")
        print("symbols in trades:",    sorted(trd['symbol'].dropna().astype(str).unique().tolist()) if 'symbol' in trd.columns else "(no symbol column)")

        # --- Diagnostics: per-symbol contribution
        if 'symbol' in dec.columns:
            print("[Diag] decisions by symbol:", dec['symbol'].value_counts().to_dict())
        if 'symbol' in trd.columns:
            print("[Diag] trades by symbol:", trd['symbol'].value_counts().to_dict())

        def _attach_y_open_to_open(decisions: pd.DataFrame, bars: pd.DataFrame, H: int) -> pd.DataFrame:
            if decisions.empty:
                return decisions
            dec = decisions.copy()
            dec["timestamp"] = pd.to_datetime(dec["timestamp"], utc=True)

            b = bars.copy()

            # ensure symbol exists
            if "symbol" not in b.columns:
                if "symbol" in dec.columns:
                    uniq = dec["symbol"].dropna().unique()
                    if len(uniq) == 1:
                        b["symbol"] = uniq[0]
                    else:
                        raise KeyError("bars missing 'symbol' and decisions has multiple symbols.")
                else:
                    raise KeyError("bars missing 'symbol' and decisions has no symbol column.")

            if b.empty:
                return dec

            b["timestamp"] = pd.to_datetime(b["timestamp"], utc=True)
            b = b.sort_values(["symbol", "timestamp"]).copy()

            b["open_fwd"] = b.groupby("symbol")["open"].shift(-H)
            b["logret_oo"] = np.log(b["open_fwd"] / b["open"])

            dec = dec.merge(b[["timestamp", "symbol", "logret_oo"]], on=["timestamp", "symbol"], how="left")
            dec["y"] = (dec["logret_oo"] > 0.0).astype(float)
            return dec


        # collect bars covering [start,end] for all symbols (cheap: only timestamp+open)
        bars_list = []
        for s in symbols:
            df_s = _load_bars_for_symbol({"parquet_root": str(parquet_root), "start": cfg["start"], "end": cfg["end"]},
                                         s)
            if not df_s.empty:
                bars_list.append(df_s[["timestamp", "symbol", "open"]])
        bars_all = pd.concat(bars_list, ignore_index=True) if bars_list else pd.DataFrame()

        if 'y' not in dec.columns and not bars_all.empty:
            dec = _attach_y_open_to_open(decisions=dec, bars=bars_all, H=int(cfg.get("horizon_bars", 20)))
            # persist back to portfolio/ so later steps see it

            # --- optional AUC-based sign check ---
            if bool(cfg.get("sign_check", False)):
                dec, flipped = _maybe_flip_probs(dec, p_col="p_cal", y_col="y")
                if flipped:
                    print("[SignCheck] p_cal flipped at portfolio diagnostics stage.")

            dec.to_parquet(port_dir / "decisions.parquet", index=False)

        # --- Diagnostics: scanner yield already printed per symbol in _run_one_symbol via [Scanner KPI]
        # Add a guard to force at least some RRC decisions for debugging
        if {'timestamp', 'symbol'}.issubset(dec.columns):
            sym_ct = dec.groupby('symbol')['timestamp'].nunique()
            print("[Diag] minutes per symbol:", sym_ct.to_dict())

        # --- Brier score / log-loss (entries only if you tag entries; otherwise whole dec file)
        try:
            from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
            d = dec.dropna(subset=['p_cal', 'y'])
            if len(d) > 10 and d['y'].nunique() == 2:
                print("[Diag] Brier:", float(brier_score_loss(d['y'], d['p_cal'])))
                print("[Diag] LogLoss:", float(log_loss(d['y'], d['p_cal'], labels=[0, 1])))
                print("[Diag] ROC AUC(all):", float(roc_auc_score(d['y'], d['p_cal'])))
        except Exception as _e:
            print(f"[Diag] Skipped prob-metric calc: {_e!r}")

        # --- EV vs p curve (binning)
        try:
            dd = dec.copy()
            if {'p_cal', 'y'}.issubset(dd.columns):
                dd['p_bin'] = pd.qcut(dd['p_cal'], q=10, duplicates='drop')
                g = dd.groupby('p_bin').agg(p_mean=('p_cal', 'mean'),
                                            y_rate=('y', 'mean'),
                                            n=('y', 'size'))
                print("[Diag] EV-by-decile:\n", g)
        except Exception as _e:
            print(f"[Diag] Skipped EV-by-decile: {_e!r}")

        # unified clock sanity: timestamps with >=2 symbols
        if {'timestamp','symbol'}.issubset(dec.columns):
            multi = (dec.groupby('timestamp')['symbol'].nunique() >= 2).mean()
            print("share of timestamps with >=2 symbols present:", round(float(multi), 3))
            # Phase 1.1 hard gate: do not proceed to portfolio artifacts if overlap is too low
            min_share = float(cfg.get("min_share_multisymbol", 0.10))
            if float(multi) < min_share:
                raise RuntimeError(
                    f"[GATE] multi-symbol share < {min_share:.2f} "
                    f"(got {float(multi):.3f}). Refusing to emit/accept portfolio artifacts."
                )

        # optional: decision→entry causality check (if you persist these)
        if {'decision_ts','entry_ts'}.issubset(dec.columns):
            lag_ok = (pd.to_datetime(dec['entry_ts']) > pd.to_datetime(dec['decision_ts'])).mean()
            print("entry strictly after decision:", round(float(lag_ok), 3))
            # Phase 4.7 extra soft checks
            port_dir = _resolve_path(cfg.get("artifacts_root", "artifacts/a2"), create=True, is_dir=True) / "portfolio"
            metrics_json = port_dir / "portfolio_metrics.json"
            print("has portfolio_metrics.json:", metrics_json.exists())
            #if (port_dir / "equity_curve.csv").exists():
            #    ec = pd.read_csv(port_dir / "equity_curve.csv", parse_dates=["exit_ts"])
            #    print("equity curve rows:", len(ec))
            if (port_dir / "equity_curve.csv").exists():
                ec = pd.read_csv(port_dir / "equity_curve.csv", parse_dates=["timestamp"])
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



__all__ = ["run", "run_batch", "CONFIG"]

#if __name__ == "__main__":
#    asyncio.run(run(CONFIG))


if __name__ == "__main__":
    asyncio.run(run(CONFIG))

    from pathlib import Path

    from feature_engineering.utils.consolidate_decisions import consolidate_decisions
    from feature_engineering.utils.walkforward_report import generate_walkforward_report
    from feature_engineering.utils.consistency_gate import ConsistencyGateInputs, run_consistency_gate

    #run_id = RUN_ID
    #artifacts_root = Path(CONFIG["artifacts_root"])

    run_id = RUN_ID

    # Phase 1.1: single canonical artifacts root for the whole run
    base_root = resolve_artifacts_root(CONFIG, create=True)  # => <repo_root>/artifacts
    run_dir = resolve_artifacts_root({"artifacts_root": str(base_root)}, run_id=RUN_ID, create=True)
    # run_dir => <repo_root>/artifacts/a2_<RUN_ID>

    # Hard gate: must never include scripts/artifacts
    if "scripts" in [p.lower() for p in run_dir.parts] and "artifacts" in [p.lower() for p in run_dir.parts]:
        s = str(run_dir).lower().replace("/", "\\")
        if "\\scripts\\artifacts\\" in s:
            raise RuntimeError(f"[Phase 1.1] Illegal artifacts root (scripts/artifacts): {run_dir}")

    # Single source of truth downstream
    CONFIG["artifacts_root"] = str(run_dir)

    print(f"[RunContext] artifacts_root={run_dir}")  # <-- the ONE line you keep

    # Required observable artifact
    (run_dir / "_ARTIFACTS_ROOT.txt").write_text(str(run_dir.resolve()), encoding="utf-8")

    import json
    from datetime import datetime, timezone

    u = CONFIG.get("universe")
    if hasattr(u, "symbols"):
        u_ser = list(u.symbols)
    elif isinstance(u, (list, tuple)):
        u_ser = list(u)
    else:
        u_ser = str(u)

    run_context = {
        "run_id": run_id,
        "artifacts_root": str(run_dir.resolve()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        # optional but useful:
        "window": {"start": str(CONFIG.get("start")), "end": str(CONFIG.get("end"))},
        "universe": CONFIG.get("universe"),
        "git": CONFIG.get("git", None),
    }
    (run_dir / "run_context.json").write_text(json.dumps(run_context, indent=2), encoding="utf-8")

    # Source of truth: the run directory chosen by RunContext, persisted at run start
    # (never trust CONFIG["artifacts_root"] if it could be a base dir or scripts/artifacts)
    artifacts_root = None
    try:
        import json
        # CONFIG["artifacts_root"] should be the run dir AFTER Task 3, but we still read run_context.json
        # as the definitive “this run wrote here” artifact.
        candidate = Path(CONFIG["artifacts_root"]).resolve()
        rc = candidate / "run_context.json"
        if rc.exists():
            artifacts_root = Path(json.loads(rc.read_text(encoding="utf-8"))["artifacts_root"]).resolve()
    except Exception:
        artifacts_root = None

    if artifacts_root is None:
        artifacts_root = Path(CONFIG["artifacts_root"]).resolve()


    # Hard fail: consolidation/reporting must run against a single run directory, not a base folder.
    if (artifacts_root / "run_context.json").exists() is False:
        raise RuntimeError(
            "[Task3] Refusing to consolidate/report without run_context.json in artifacts_root. "
            f"Got artifacts_root={artifacts_root}"
        )


    searched_dirs = [
        artifacts_root,
        artifacts_root / "phases",
        artifacts_root / "backtests",
    ]

    cons = consolidate_decisions(
        run_id=run_id,
        artifacts_root=artifacts_root,
        searched_dirs=searched_dirs,
        verbose=True,
    )
    rep = generate_walkforward_report(
        run_id=run_id,
        artifacts_root=artifacts_root,
        searched_dirs=searched_dirs,
        verbose=True,
    )

    run_consistency_gate(
        ConsistencyGateInputs(
            run_id=run_id,
            artifacts_root=artifacts_root,
            searched_dirs=tuple(searched_dirs),

            decisions_paths_found=len(cons.decision_files_found),
            consolidated_decisions_written=(cons.consolidated_path is not None),
            consolidated_decisions_rows=cons.rows_written,

            report_phase_decisions_count=rep.decisions_count_seen,
            report_said_no_decisions=rep.report_had_no_decisions_banner,

            decision_files_found=cons.decision_files_found,
            consolidated_path=cons.consolidated_path,
            report_path=rep.report_path,
        )
    )
