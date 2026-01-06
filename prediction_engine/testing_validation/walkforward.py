# prediction_engine/testing_validation/walkforward.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any
from feature_engineering.utils.time import to_utc

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
# after: from prediction_engine.tx_cost import BasicCostModel
from prediction_engine.market_regime import MarketRegime  # <-- ADD

from feature_engineering.pipelines.core import CoreFeaturePipeline
from prediction_engine.ev_engine import EVEngine
from prediction_engine.calibration import calibrate_isotonic, map_mu_to_prob
from prediction_engine.prediction_engine.artifacts.loader import resolve_artifact_paths, load_calibrator, read_distance_contract

import shutil
import platform
import sys
import sklearn

from prediction_engine.tx_cost import BasicCostModel


@dataclass(frozen=True)
class Fold:
    idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    purge_bars: int



def _ts_diag(name: str, x) -> str:
    """Return a compact diagnostic string for timestamp-like things."""
    import pandas as pd

    if x is None:
        return f"[{name}] <None>"

    # Accept Series / Index / list-like
    s = pd.Series(x) if not isinstance(x, (pd.Series, pd.Index)) else x
    if isinstance(s, pd.Index):
        s = pd.Series(s)

    out = [f"[{name}]"]
    out.append(f"  dtype: {getattr(s, 'dtype', type(s))}")
    # tz info (works if datetime64[ns, tz])
    try:
        tz = getattr(getattr(s.dt, "tz", None), "zone", None) or getattr(s.dt, "tz", None)
    except Exception:
        tz = None
    out.append(f"  tz: {tz}")

    # head/tail samples
    try:
        head = s.head(3).tolist()
        tail = s.tail(3).tolist()
        out.append(f"  head: {head}")
        out.append(f"  tail: {tail}")
    except Exception:
        pass

    # min/max
    try:
        s_dt = pd.to_datetime(s, utc=True, errors="coerce")
        s_dt = s_dt.dropna()
        if len(s_dt):
            out.append(f"  min/max: {s_dt.min()} → {s_dt.max()}")
        else:
            out.append("  min/max: <empty>")
    except Exception:
        out.append("  min/max: <unavailable>")

    return "\n".join(out)


def _audit_scan_fold_intersection(
    *,
    full_df,
    scan_df,
    full_tr,
    full_te,
    scan_tr,
    scan_te,
    allow_fallback: bool,
    fold_idx: int,
) -> tuple[list, list]:
    """
    Audit BEFORE/AFTER scan↔fold timestamp intersection.
    Hard-fail if scan has rows but both intersections are empty (unless allow_fallback).
    Returns: (scan_tr_ts_in_fold, scan_te_ts_in_fold) as python lists of timestamps.
    """
    import pandas as pd

    # ---- BEFORE ----
    print(f"[Fold {fold_idx}] [SCAN-AUDIT] BEFORE intersection")
    print(_ts_diag("FULL.timestamp", full_df["timestamp"]))
    print(_ts_diag("SCAN.timestamp", scan_df["timestamp"]))
    print(_ts_diag("FULL_TR.timestamp", full_tr["timestamp"]))
    print(_ts_diag("FULL_TE.timestamp", full_te["timestamp"]))
    print(_ts_diag("SCAN_TR.timestamp", scan_tr["timestamp"]))
    print(_ts_diag("SCAN_TE.timestamp", scan_te["timestamp"]))
    print(f"[Fold {fold_idx}] [SCAN-AUDIT] counts: scan_tr={len(scan_tr)} scan_te={len(scan_te)}")

    # Normalize to UTC for safe intersection (handles tz-aware vs naive mismatches)
    def _utc_series(df):
        return pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    tr_full_ts = pd.Index(_utc_series(full_tr).dropna())
    te_full_ts = pd.Index(_utc_series(full_te).dropna())
    tr_scan_ts = pd.Index(_utc_series(scan_tr).dropna())
    te_scan_ts = pd.Index(_utc_series(scan_te).dropna())

    tr_in = tr_scan_ts.intersection(tr_full_ts)
    te_in = te_scan_ts.intersection(te_full_ts)

    # ---- AFTER ----
    print(f"[Fold {fold_idx}] [SCAN-AUDIT] AFTER intersection")
    print(f"[Fold {fold_idx}] [SCAN-AUDIT] len(scan_tr_in_fold)={len(tr_in)} len(scan_te_in_fold)={len(te_in)}")
    if len(tr_in):
        print(f"[Fold {fold_idx}] [SCAN-AUDIT] scan_tr_in_fold min/max: {tr_in.min()} → {tr_in.max()}")
    if len(te_in):
        print(f"[Fold {fold_idx}] [SCAN-AUDIT] scan_te_in_fold min/max: {te_in.min()} → {te_in.max()}")

    # ---- HARD FAIL on contradiction ----
    if (len(scan_tr) > 0 or len(scan_te) > 0) and (len(tr_in) == 0 and len(te_in) == 0) and (not allow_fallback):
        msg = (
            f"[Fold {fold_idx}] SCAN↔FOLD CONTRADICTION: scan_tr/scan_te are nonzero "
            f"(scan_tr={len(scan_tr)}, scan_te={len(scan_te)}) but both intersections are 0.\n"
            f"{_ts_diag('FULL.timestamp', full_df['timestamp'])}\n"
            f"{_ts_diag('SCAN.timestamp', scan_df['timestamp'])}\n"
            f"{_ts_diag('FULL_TR.timestamp', full_tr['timestamp'])}\n"
            f"{_ts_diag('FULL_TE.timestamp', full_te['timestamp'])}\n"
            f"{_ts_diag('SCAN_TR.timestamp', scan_tr['timestamp'])}\n"
            f"{_ts_diag('SCAN_TE.timestamp', scan_te['timestamp'])}\n"
            "This would previously have caused silent fallback. Refusing to continue."
        )
        raise RuntimeError(msg)

    # Return python lists (useful for isin)
    return list(tr_in), list(te_in)


# --- INSERT AFTER: the Fold dataclass definition ---

def make_time_folds(
    *,
    window_start: "pd.Timestamp | str",
    window_end: "pd.Timestamp | str",
    mode: str,
    n_folds: int,
    test_span_days: int,
    train_min_days: int,
    embargo_days: int,
) -> list[Fold]:
    """
    Build time-only folds with an explicit embargo gap between TRAIN and TEST.
      • Expanding: TRAIN starts at window_start; TRAIN end is (test_start - embargo)
      • Rolling:   TRAIN has fixed length; TRAIN end is (test_start - embargo)
      • TEST:      [test_start, test_start + test_span_days)  (left-closed, right-open)
    Returns a list of Fold(idx, train_start, train_end, test_start, test_end, purge_bars=embargo_days).
    NOTE: We store embargo_days in the Fold.purge_bars field to avoid changing the dataclass shape.
    """
    import pandas as _pd

    #ws = _pd.to_datetime(window_start)
    #we = _pd.to_datetime(window_end)

    #ws = _pd.to_datetime(window_start, utc=True)
    #we = _pd.to_datetime(window_end, utc=True)

    ws = to_utc(window_start)
    we = to_utc(window_end)

    if not (ws < we):
        raise ValueError("window_start must be < window_end")

    mode = str(mode).lower()
    if mode not in {"expanding", "rolling"}:
        raise ValueError("mode must be 'expanding' or 'rolling'")

    if n_folds <= 0:
        raise ValueError("n_folds must be >= 1")
    if test_span_days <= 0 or train_min_days <= 0 or embargo_days < 0:
        raise ValueError("test_span_days > 0, train_min_days > 0, embargo_days >= 0 required")

    # Compute test starts uniformly over the window
    total_days = (we.normalize() - ws.normalize()).days
    if total_days < (train_min_days + embargo_days + test_span_days):
        raise ValueError("Window too short for requested train/embargo/test lengths.")

    # Even spacing of test blocks (not overlapping), clipped to fit inside the window
    # We’ll choose test_start candidates and take the first n_folds that fit.
    # TEST is left-closed/right-open to avoid boundary double-count.
    test_starts = []
    cursor = ws.normalize() + _pd.Timedelta(days=train_min_days + embargo_days)
    while cursor + _pd.Timedelta(days=test_span_days) <= we:
        test_starts.append(cursor)
        # Next test block starts right after the previous test block
        cursor = cursor + _pd.Timedelta(days=test_span_days)

    if len(test_starts) == 0:
        raise ValueError("Could not place any test folds with given constraints.")
    if len(test_starts) > n_folds:
        test_starts = test_starts[:n_folds]

    folds: list[Fold] = []
    for i, ts in enumerate(test_starts, start=1):
        te = ts + _pd.Timedelta(days=test_span_days)  # right-open
        if te > we:
            break  # don’t create ragged partial fold beyond the window

        # Train end is test_start - embargo
        train_end = ts - _pd.Timedelta(days=embargo_days)
        if mode == "expanding":
            train_start = ws
        else:  # rolling
            train_start = train_end - _pd.Timedelta(days=train_min_days)
            if train_start < ws:
                # not enough space for a full rolling window; skip
                continue

        if not (train_start < train_end and ts < te):
            # Skip degenerate folds
            continue

        '''folds.append(
            Fold(
                idx=i,
                train_start=train_start.tz_localize(None) if hasattr(train_start, "tz") else train_start,
                train_end=(train_end - _pd.Timedelta(nanoseconds=1)),  # make TRAIN right-closed
                test_start=ts,
                test_end=(te - _pd.Timedelta(nanoseconds=1)),         # make TEST right-closed for persistence
                purge_bars=int(embargo_days),  # store embargo_days here for compatibility
            )
        )'''

        '''folds.append(
            Fold(
                idx=i,
                train_start=_pd.Timestamp(train_start, tz="UTC") if getattr(train_start, "tz",
                                                                            None) is None else train_start,
                train_end=(train_end - _pd.Timedelta(nanoseconds=1)),
                test_start=_pd.Timestamp(ts, tz="UTC") if getattr(ts, "tz", None) is None else ts,
                test_end=(te - _pd.Timedelta(nanoseconds=1)),
                purge_bars=int(embargo_days),
            )
        )'''

        folds.append(
            Fold(
                idx=i,
                train_start=to_utc(train_start),
                train_end=to_utc(train_end - _pd.Timedelta(nanoseconds=1)),
                test_start=to_utc(ts),
                test_end=to_utc(te - _pd.Timedelta(nanoseconds=1)),
                purge_bars=int(embargo_days),
            )
        )

    if not folds:
        raise ValueError("No valid folds produced. Check your time bounds and parameters.")
    return folds


def save_folds_json(artifacts_root: "Path | str", folds: list[Fold]) -> "Path":
    """
    Write folds.json under artifacts_root/folds/
    """
    root = Path(artifacts_root)
    outdir = root / "folds"
    outdir.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "idx": f.idx,
            "train_start": str(f.train_start),
            "train_end": str(f.train_end),
            "test_start": str(f.test_start),
            "test_end": str(f.test_end),
            "embargo_days": int(f.purge_bars),  # stored here
        }
        for f in folds
    ]
    out = outdir / "folds.json"
    out.write_text(json.dumps(payload, indent=2))
    return out




class WalkForwardRunner:
    """
    A2 (revised): Purged walk-forward built off the FULL bar stream.
      • Folds/purge/labels computed on df_full (unscanned).
      • Scanner-filtered rows (df_scanned) are the only bars evaluated for decisions.
      • Fit FE & EV artefacts on TRAIN window only; transform/evaluate on TEST.
      • Train isotonic on TRAIN; compute gates from TRAIN; apply to TEST.
    Artifacts per fold:
      artifacts_root/fold_{i}/
        calibration/iso_calibrator.pkl
        gates.json
        decisions.parquet
        metrics.json
    """

    def __init__(
        self,
        *,
        artifacts_root: Path,
        parquet_root: Path,
        ev_artifacts_root: Path,
        symbol: str,
        horizon_bars: int,
        longest_lookback_bars: int,
        p_gate_q: float = 0.65,
        full_p_q: float = 0.80,
        tz: str = "America/New_York",
        debug_no_costs: bool = False,

    ) -> None:
        self.artifacts_root = Path(artifacts_root)
        self.parquet_root = Path(parquet_root)
        self.ev_artifacts_root = Path(ev_artifacts_root)
        self.symbol = symbol
        self.horizon_bars = int(horizon_bars)
        self.longest_lookback_bars = int(longest_lookback_bars)
        self.p_gate_q = float(p_gate_q)
        self.full_p_q = float(full_p_q)
        self.tz = tz
        self.debug_no_costs = debug_no_costs

    # ------------------------------ helpers ------------------------------

    @staticmethod
    def _month_edges(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
        start = pd.Timestamp(start).normalize()
        end = pd.Timestamp(end).normalize()
        idx = pd.date_range(start=start, end=end, freq="MS")
        if len(idx) and idx[-1] < end:
            idx = idx.append(pd.DatetimeIndex([end + pd.offsets.MonthBegin()]))
        return list(idx)

    @staticmethod
    def _median_bar_delta(df: pd.DataFrame) -> pd.Timedelta:
        if len(df) < 2:
            return pd.Timedelta(minutes=1)
        dt = df["timestamp"].diff().dropna().median()
        # If scanning made bars sparse (e.g., multi-hour gaps), fall back to 1 minute
        return dt if pd.Timedelta(minutes=1) <= dt <= pd.Timedelta(hours=1) else pd.Timedelta(minutes=1)

    # ============================
    # DEBUG / DIAGNOSTICS HELPERS
    # ============================

    @staticmethod
    def _force_utc_ts(df: pd.DataFrame, col: str = "timestamp") -> pd.DataFrame:
        """
        Hard-normalize a timestamp column to datetime64[ns, UTC].
        Drops NaT and sorts by timestamp.
        """
        df = df.copy()
        if col not in df.columns:
            raise KeyError(f"Missing '{col}' column")
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
        df = df[df[col].notna()].sort_values(col).reset_index(drop=True)
        return df

    @staticmethod
    def _ts_diag(name: str, df: pd.DataFrame) -> None:
        print(f"\n[TS-DIAG] {name}")
        print("  shape:", df.shape)
        if "timestamp" not in df.columns:
            print("  MISSING timestamp COLUMN")
            return
        s = df["timestamp"]
        print("  dtype:", s.dtype)
        try:
            print("  head:", s.head(3).tolist())
            print("  tail:", s.tail(3).tolist())
        except Exception as e:
            print("  head/tail failed:", e)

        s2 = pd.to_datetime(s, errors="coerce", utc=True)
        print("  to_datetime(utc=True) dtype:", s2.dtype)
        print("  coerced NaT:", int(s2.isna().sum()))
        if s2.notna().any():
            print("  min/max:", str(s2.min()), "→", str(s2.max()))

    def _fold_diag(self, f: Fold, full: pd.DataFrame, scan: pd.DataFrame) -> None:
        print(f"\n[FOLD-DIAG] fold={f.idx}")
        print("  train:", f.train_start, "→", f.train_end, " purge_bars:", f.purge_bars)
        print("  test: ", f.test_start,  "→", f.test_end)

        for k, t in [("train_start", f.train_start), ("train_end", f.train_end),
                     ("test_start", f.test_start), ("test_end", f.test_end)]:
            try:
                print(f"  {k}: tz={getattr(t, 'tz', None)}  type={type(t)}")
            except Exception:
                print(f"  {k}: type={type(t)}")

        self._ts_diag("FULL", full)
        self._ts_diag("SCAN", scan)

        # Mask counts preview (these should NOT be zero in your case)
        bar_dt = self._median_bar_delta(full)
        purge_cut = f.purge_bars * bar_dt

        tr_mask_full = (full["timestamp"] >= f.train_start) & (full["timestamp"] <= f.train_end - purge_cut)
        te_mask_full = (full["timestamp"] >= f.test_start)  & (full["timestamp"] <= f.test_end)
        tr_mask_scan = (scan["timestamp"] >= f.train_start) & (scan["timestamp"] <= f.train_end - purge_cut)
        te_mask_scan = (scan["timestamp"] >= f.test_start)  & (scan["timestamp"] <= f.test_end)

        print("  mask counts | full_tr:", int(tr_mask_full.sum()), "full_te:", int(te_mask_full.sum()),
              "| scan_tr:", int(tr_mask_scan.sum()), "scan_te:", int(te_mask_scan.sum()))


    def _build_folds(
        self, start: pd.Timestamp, end: pd.Timestamp, df_full: pd.DataFrame
    ) -> List[Fold]:
        bar_dt = self._median_bar_delta(df_full)
        purge_bars = max(self.horizon_bars, self.longest_lookback_bars)

        edges = self._month_edges(start, end)
        folds: List[Fold] = []
        for i in range(1, len(edges)):
            test_start = edges[i]
            # next month start (or end+1M if i is the last edge)
            next_start = edges[i + 1] if i + 1 < len(edges) else (end + pd.offsets.MonthBegin(1))
            # end at the last bar strictly before next_start
            test_end = min(next_start - bar_dt, end)
            if test_start > end:
                break
            train_end = test_start - bar_dt
            train_start = start
            '''folds.append(
                Fold(
                    idx=len(folds) + 1,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    purge_bars=purge_bars,
                )
            )'''

            folds.append(
                Fold(
                    idx=len(folds) + 1,
                    train_start=to_utc(train_start),
                    train_end=to_utc(train_end),
                    test_start=to_utc(test_start),
                    test_end=to_utc(test_end),
                    purge_bars=purge_bars,
                )
            )

        return folds

    # ------------------------------ core API ------------------------------

    def run(
            self,
            *,
            df_full: pd.DataFrame,
            df_scanned: pd.DataFrame,
            start: str | pd.Timestamp,
            end: str | pd.Timestamp,
            p_gate_q: float | None = None,
            full_p_q: float | None = None,
            # NEW (optional) Phase-2 hooks:
            calibrator_fn: Optional[Callable[[np.ndarray, np.ndarray], Any]] = None,
            ev_engine_overrides: Optional[Dict[str, Any]] = None,
            persist_prob_columns: Tuple[str, str] | None = None,  # e.g., ("p_raw", "p_cal")
    ) -> Dict:

        '''start = pd.to_datetime(start)
        end = pd.to_datetime(end)

        # filter both frames to window (defensive)
        full = df_full[(df_full["timestamp"] >= start) & (df_full["timestamp"] <= end)].reset_index(drop=True)
        scan = df_scanned[(df_scanned["timestamp"] >= start) & (df_scanned["timestamp"] <= end)].reset_index(drop=True)'''

        # ============================
        # FORCE UTC TIMESTAMPS (CRITICAL)
        # ============================
        start = pd.to_datetime(start, utc=True)
        end = pd.to_datetime(end, utc=True)

        # normalize timestamp columns
        full_all = self._force_utc_ts(df_full, "timestamp")
        scan_all = self._force_utc_ts(df_scanned, "timestamp")

        # filter both frames to window (defensive)
        full = full_all[(full_all["timestamp"] >= start) & (full_all["timestamp"] <= end)].reset_index(drop=True)
        scan = scan_all[(scan_all["timestamp"] >= start) & (scan_all["timestamp"] <= end)].reset_index(drop=True)

        if full.empty:
            raise ValueError("No rows in df_full after date filter")

        # Quick window sanity
        print(f"[WINDOW] start={start} end={end}")
        print(f"[WINDOW] full rows={len(full)} | scan rows={len(scan)}")
        print(f"[WINDOW] full min/max={full['timestamp'].min()} → {full['timestamp'].max()}")
        if not scan.empty:
            print(f"[WINDOW] scan min/max={scan['timestamp'].min()} → {scan['timestamp'].max()}")


        if full.empty:
            raise ValueError("No rows in df_full after date filter")
        # Note: scan MAY be empty in a given month; we’ll still compute folds/metrics.

        # Precompute next-bar label on FULL stream (binary up on next bar)
        full = full.sort_values("timestamp").reset_index(drop=True)
        ret1_full = (full["close"].shift(-1) / full["open"].shift(-1) - 1.0)
        y_full = (ret1_full.fillna(0.0) > 0).astype(int).rename("y")
        # Align helper: quick index by timestamp
        y_by_ts = pd.Series(y_full.values, index=full["timestamp"].values)

        #folds = self._build_folds(start, end, full)

        # --- INSERT near the top of WalkForwardRunner.run(...) after you resolve start/end ---

        folds = make_time_folds(
            window_start=start,
            window_end=end,
            mode="expanding",  # or "rolling" or param
            n_folds=6,  # example
            test_span_days=20,  # example
            train_min_days=60,  # example
            embargo_days=3,  # example
        )
        save_folds_json(self.artifacts_root, folds)

        all_fold_metrics = []
        total_entries = 0
        total_rows = 0

        for f in folds:
            fold_dir = self.artifacts_root / f"fold_{f.idx:02d}"

            # Phase 1.1 guard: prevent duplicated "a2\\a2" style roots
            _fold_path_norm = str(fold_dir).lower().replace("/", "\\")
            if "\\a2\\a2\\" in _fold_path_norm:
                raise AssertionError(f"[Phase1.1] illegal duplicated a2 segment in fold_dir: {fold_dir}")

            calib_dir = fold_dir / "calibration"
            plots_dir = fold_dir / "plots"
            for d in (calib_dir, plots_dir):
                d.mkdir(parents=True, exist_ok=True)

            bar_dt = self._median_bar_delta(full)
            purge_cut = f.purge_bars * bar_dt

            # ============================
            # DEBUG: PRINT FOLD 1 ONLY
            # ============================
            if int(f.idx) == 1:
                self._fold_diag(f, full, scan)


            # TRAIN on FULL (with purge)
            tr_mask_full = (full["timestamp"] >= f.train_start) & (full["timestamp"] <= f.train_end - purge_cut)
            te_mask_full = (full["timestamp"] >= f.test_start) & (full["timestamp"] <= f.test_end)
            full_tr = full.loc[tr_mask_full].reset_index(drop=True)
            full_te = full.loc[te_mask_full].reset_index(drop=True)
            if full_tr.empty or full_te.empty:
                # Even if empty, write a fold record for visibility
                fold_dir = self.artifacts_root / f"fold_{f.idx:02d}"
                (fold_dir / "calibration").mkdir(parents=True, exist_ok=True)
                (fold_dir / "plots").mkdir(parents=True, exist_ok=True)
                scan_rows = int(((scan["timestamp"] >= f.test_start) & (scan["timestamp"] <= f.test_end)).sum())

                print(
                    f"[Fold {f.idx}] EMPTY slice. "
                    f"full_tr={len(full_tr)} full_te={len(full_te)} "
                    f"| scan_rows(test)={scan_rows} "
                    f"| full_ts_dtype={full['timestamp'].dtype} scan_ts_dtype={scan['timestamp'].dtype} "
                    f"| fold_test={f.test_start}→{f.test_end}"
                )


                m = {
                    "fold": f.idx,
                    "train_window": [str(f.train_start), str(f.train_end)],
                    "test_window": [str(f.test_start), str(f.test_end)],
                    "purge_bars": f.purge_bars,
                    "auc_all": float("nan"),
                    "auc_on_entries": float("nan"),
                    "n_test": scan_rows,
                    "n_entries": 0,
                    "p_gate": float("nan"),
                    "full_p": float("nan"),
                    "reliability": {"p_mean": [], "y_rate": [], "n": []},
                }
                (fold_dir / "metrics.json").write_text(json.dumps(m, indent=2))
                all_fold_metrics.append(m)
                continue

            # SCANNED subsets per window (may be empty in either window)
            tr_mask_scan = (scan["timestamp"] >= f.train_start) & (scan["timestamp"] <= f.train_end - purge_cut)
            te_mask_scan = (scan["timestamp"] >= f.test_start) & (scan["timestamp"] <= f.test_end)
            scan_tr = scan.loc[tr_mask_scan].reset_index(drop=True)
            scan_te = scan.loc[te_mask_scan].reset_index(drop=True)

            # ---- Phase 2: scan→fold intersection audit + hard-fail on contradiction ----
            allow_scan_fallback = bool(getattr(self, "allow_scan_fallback", False))  # dev-only escape hatch

            scan_tr_in_fold_ts, scan_te_in_fold_ts = _audit_scan_fold_intersection(
                full_df=full,
                scan_df=scan,
                full_tr=full_tr,
                full_te=full_te,
                scan_tr=scan_tr,
                scan_te=scan_te,
                allow_fallback=allow_scan_fallback,
                fold_idx=f.idx,
            )

            # ---- FE: fit on full TRAIN, transform SCANNED train/test (no eval fit)
            '''pipe = CoreFeaturePipeline(parquet_root=Path(""))
            feats_full_tr, _ = pipe.run_mem(full_tr)       # fits scaler/PCA
            feats_scan_tr = pipe.transform_mem(scan_tr) if not scan_tr.empty else pd.DataFrame()
            feats_scan_te = pipe.transform_mem(scan_te) if not scan_te.empty else pd.DataFrame()
            '''
            # ---- FE: fit on full TRAIN, transform full TEST, then filter for scanned bars
            '''pipe = CoreFeaturePipeline(parquet_root=Path(""))

            # 1. Run full feature engineering AND fit scalers/PCA on the full training data.
            #    This "teaches" the pipeline the data distributions from the train set.
            print(f"[Fold {f.idx}] Fitting FE pipeline on {len(full_tr)} full train bars...")
            feats_full_tr, _ = pipe.run_mem(full_tr)
            '''

            # ---- FE: fit on full TRAIN, transform full TEST, then filter for scanned bars
            prepro_dir = fold_dir / "prepro"
            prepro_dir.mkdir(parents=True, exist_ok=True)

            # Make the pipeline write its fitted scaler/pca under the fold's prepro dir
            from pathlib import Path as _Path
            pipe = CoreFeaturePipeline(parquet_root=_Path(prepro_dir))

            # 1. Run full feature engineering AND fit scalers/PCA on the full training data.
            print(f"[Fold {f.idx}] Fitting FE pipeline on {len(full_tr)} full train bars...")
            feats_full_tr, _ = pipe.run_mem(full_tr)  # this will dump scaler.pkl/pca.pkl into prepro_dir/_fe_meta

            # Save the feature list (PCA columns) used by this fold
            pc_cols = [c for c in feats_full_tr.columns if c.startswith("pca_")]
            (prepro_dir / "feature_cols.json").write_text(json.dumps(pc_cols))

            # 2. Now, use the FITTED pipeline to transform the full test data.
            #    This applies the learned scaling/PCA without leaking information from the test set.
            print(f"[Fold {f.idx}] Transforming {len(full_te)} full test bars...")
            feats_full_te = pipe.transform_mem(full_te) if not full_te.empty else pd.DataFrame()

            # 3. Instead of re-calculating, just SELECT the rows we need from the results above.
            #    This is efficient and guarantees the columns exist.
            '''feats_scan_tr = feats_full_tr.loc[feats_full_tr['timestamp'].isin(scan_tr['timestamp'])].reset_index(
                drop=True)
            feats_scan_te = feats_full_te.loc[feats_full_te['timestamp'].isin(scan_te['timestamp'])].reset_index(
                drop=True)'''

            feats_scan_tr = feats_full_tr.loc[
                pd.to_datetime(feats_full_tr["timestamp"], utc=True, errors="coerce").isin(scan_tr_in_fold_ts)
            ].reset_index(drop=True)

            feats_scan_te = feats_full_te.loc[
                pd.to_datetime(feats_full_te["timestamp"], utc=True, errors="coerce").isin(scan_te_in_fold_ts)
            ].reset_index(drop=True)

            print(
                f"[Fold {f.idx}] Found {len(feats_scan_tr)} scanned train bars and {len(feats_scan_te)} scanned test bars.")

            # ---- Build per-row regime series aligned by timestamp -----------------
            def _aligned_regime_series(base_df: pd.DataFrame, feats_df: pd.DataFrame,
                                       fallback: str = "TREND") -> pd.Series:
                """
                Returns a string regime series aligned to feats_df['timestamp'].
                If base_df has no 'regime' column, everything defaults to 'TREND'.
                """
                if "regime" in base_df.columns:
                    ser = (base_df[["timestamp", "regime"]]
                    .dropna()
                    .drop_duplicates(subset=["timestamp"])
                    .set_index("timestamp")["regime"])
                    out = ser.reindex(feats_df["timestamp"]).fillna(fallback).astype(str)
                else:
                    out = pd.Series(fallback, index=feats_df.index)
                return out

            reg_tr = _aligned_regime_series(scan_tr, feats_scan_tr, fallback="TREND")
            reg_te = _aligned_regime_series(scan_te, feats_scan_te, fallback="TREND")

            # ---- EV artefacts: build on TRAIN window only
            from scripts.rebuild_artefacts import rebuild_if_needed

            rebuild_if_needed(
                artefact_dir=str(self.ev_artifacts_root),
                parquet_root=str(self.parquet_root),
                symbols=[self.symbol],
                start=f.train_start,
                end=f.train_end - purge_cut,
                n_clusters=64,
                fitted_pipeline_dir=prepro_dir,  # ← ensures SAME PCA basis
            )

            # --- 3.6: resolve artefact paths & distance contract for traceability ---
            paths = resolve_artifact_paths(
                artifacts_root=self.ev_artifacts_root,
                symbol=self.symbol,
                strategy="pooled",  # or "per_symbol" if you run that mode in this runner
            )
            dist_family, dist_params = read_distance_contract(paths["meta"])

            # Optional: load symbol calibrator resolved by the loader (overrides fold-trained iso if present)
            #cal_from_store = load_calibrator(paths.get("calibrator", ""))
            # prefer fold-trained calibrator if it exists; else use stored symbol calibrator

            # Optional: load symbol calibrator resolved by the loader (overrides fold-trained iso if present)
            cal_from_store = load_calibrator(paths.get("calibrator", ""))

            # Ensure 'iso' exists even if no fold-trained calibrator is produced in this run
            iso = locals().get("iso", None)

            calibrator_for_fold = iso if iso is not None else cal_from_store

            #ev_tr = EVEngine.from_artifacts(self.ev_artifacts_root)
            # Decide cost handling once and pass it into both train/test engines
            cost_model = None if self.debug_no_costs else BasicCostModel()
            #ev_tr = EVEngine.from_artifacts(self.ev_artifacts_root, cost_model=cost_model)
            ev_tr = EVEngine.from_artifacts(self.ev_artifacts_root, cost_model=cost_model)

            # Snapshot the EV artefacts that were just (re)built for this fold
            ev_src = Path(self.ev_artifacts_root)
            ev_dst = fold_dir / "ev"
            ev_dst.mkdir(parents=True, exist_ok=True)
            for p in ev_src.iterdir():
                if p.is_file():
                    shutil.copy2(p, ev_dst / p.name)

            # ---- 5.3: run fold TEST via backtest callable ------------------------------
            from scripts.run_backtest import run_batch as run_bt

            # --- execution rules required by scripts.run_backtest.run_batch ---
            # Keep it minimal + compatible with your current simulator.
            execution_rules = self.__dict__.get("execution_rules") or {
                "max_fill_frac_of_bar": 1.0,
                "allow_partial": True,
            }

            test_cfg = {
                "parquet_root": str(self.parquet_root),
                "universe": [self.symbol],  # or a list of symbols in your universe
                "start": str(pd.to_datetime(f.test_start).date()),
                "end": str(pd.to_datetime(f.test_end).date()),
                "equity": float(self.__dict__.get("equity", 100_000.0)),
                "commission": 0.005,
                "debug_no_costs": bool(self.debug_no_costs),
                "vectorized_scoring": True,
                # carry over fold FE prepro dir if you saved it above
                "prepro_dir": str(prepro_dir),
                # optional: if you have a ready test features frame:
                # "test_features_df": feats_scan_te,
                "horizon_bars": int(self.horizon_bars),
                "execution_rules": execution_rules,

            }

            test_cfg["is_fold_run"] = True
            # fold runs are often single-symbol, so never enforce multi-symbol overlap here
            test_cfg["phase11_enforce_multisymbol_overlap_gate"] = False

            bt_metrics = run_bt(test_cfg, artifacts_dir=fold_dir, ev_artifacts_dir=ev_dst)

            # Sanity gate for entries (helps acceptance)
            min_entries = int(self.__dict__.get("min_entries_per_fold", 50))
            n_tr = int(bt_metrics.get("n_trades", 0))
            if n_tr < min_entries:
                print(f"[Fold {f.idx}] WARNING: only {n_tr} trades (< {min_entries}).")

            # Try to load EV meta for manifest
            ev_meta = {}
            try:
                ev_meta_path = ev_src / "meta.json"
                if ev_meta_path.exists():
                    ev_meta = json.loads(ev_meta_path.read_text())
            except Exception as _e:
                ev_meta = {"error": f"meta read failed: {type(_e).__name__}"}

            '''def _mu_on_feats(ev: EVEngine, feats_df: pd.DataFrame) -> np.ndarray:
                if feats_df.empty:
                    return np.array([], dtype=float)
                pc_cols = [c for c in feats_df.columns if c.startswith("pca_")]
                X = feats_df[pc_cols].to_numpy(dtype=np.float32)
                # --- FIX: Access the .mu attribute by name ---
                return np.array([ev.evaluate(x).mu for x in X], dtype=float)
            '''

            # ---- 5.4: per-fold metrics (calibration, analog fidelity, residual QC) ----
            from prediction_engine.testing_validation.fold_metrics import (
                compute_and_save_fold_metrics, FoldMetricsConfig
            )

            metrics_dir = fold_dir / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)

            compute_and_save_fold_metrics(
                decisions_parquet=fold_dir / "decisions.parquet",
                trades_parquet=fold_dir / "trades.parquet",
                out_dir=metrics_dir,
                cfg=FoldMetricsConfig(
                    p_col="p_cal",
                    regime_col="regime",
                    side_col="side",
                    decision_ts_col="timestamp",
                    symbol_col="symbol",
                    trade_id_col="decision_id",  # leave as-is; function will fallback if missing
                    entry_ts_col="entry_ts",
                    realized_pnl_col="realized_pnl_after_costs",
                    nn_mu_col_candidates=("nn_mu", "nn_expected_outcome", "mu_hist"),
                    fallback_col_candidates=("fallback", "used_global", "used_default"),
                    dpi=110,
                ),
                # Optional PSI inputs (provide if you captured samples earlier in 5.2/5.3)
                train_feature_samples=None,
                test_feature_samples=None,
                psi_features=None,
            )

            def _mu_on_feats(ev: EVEngine, feats_df: pd.DataFrame, regimes: Optional[pd.Series] = None) -> np.ndarray:
                """
                Evaluate EV for each row in feats_df, optionally with a per-row regime
                (aligned by feats_df['timestamp']). If no regime is provided, default TREND.
                """
                if feats_df.empty:
                    return np.array([], dtype=float)

                pc_cols = [c for c in feats_df.columns if c.startswith("pca_")]
                X = feats_df[pc_cols].to_numpy(dtype=np.float32)

                # Build an aligned regime list
                if regimes is not None and len(regimes) == len(feats_df):
                    reg_strings = regimes.astype(str).tolist()
                else:
                    reg_strings = ["TREND"] * len(feats_df)

                out = []
                for x, rstr in zip(X, reg_strings):
                    try:
                        reg = MarketRegime.from_string(rstr)  # if your enum has this
                    except Exception:
                        # Fallbacks for common enum styles
                        try:
                            reg = MarketRegime[rstr.upper()]
                        except Exception:
                            reg = MarketRegime.TREND
                    out.append(ev.evaluate(x, regime=reg).mu)
                return np.array(out, dtype=float)

            #mu_tr = _mu_on_feats(ev_tr, feats_scan_tr)
            mu_tr = _mu_on_feats(ev_tr, feats_scan_tr, regimes=reg_tr)

            # --- Calculate y_tr BEFORE you try to use it ---
            if not scan_tr.empty and "timestamp" in scan_tr.columns:
                y_tr = y_by_ts.reindex(scan_tr["timestamp"].values).fillna(0).astype(int).to_numpy()
            else:
                y_tr = np.array([], dtype=int)

            # If no scanned train rows, fall back to full train for calibration
            if mu_tr.size == 0 and not full_tr.empty:
                print(f"[Fold {f.idx}] No scanned train data; falling back to full train data for calibration.")
                feats_full_tr_scan = pipe.transform_mem(full_tr)
                mu_tr = _mu_on_feats(ev_tr, feats_full_tr_scan)
                y_tr = y_full.loc[full_tr.index].to_numpy()

            # after building feats_scan_tr / mu_tr / y_tr
            MIN_CAL_N = 100  # or 50 if your data is sparse

            use_full_for_cal = (
                    mu_tr.size < MIN_CAL_N or
                    len(np.unique(y_tr)) < 2
            )
            if use_full_for_cal:
                print(f"[Fold {f.idx}] Too few scanned train rows "
                      f"(n={mu_tr.size}, uniqY={len(np.unique(y_tr))}); "
                      f"using FULL train for calibration & gates.")
                # transform FULL train with the same fitted pipeline
                feats_full_tr_scan = pipe.transform_mem(full_tr)
                mu_tr = _mu_on_feats(ev_tr, feats_full_tr_scan)
                y_tr = y_full.loc[full_tr.index].to_numpy()

            if mu_tr.size and len(np.unique(y_tr)) > 1:
                corr = float(np.corrcoef(mu_tr, y_tr)[0, 1])
                print(f"[Fold {f.idx}] Train corr(mu, y)={corr:+.3f}, y_pos_rate={y_tr.mean():.3f}")

            # ---- TRAIN calibration with robust fallback ----
            iso = None  # Default to no calibrator
            if len(np.unique(y_tr)) > 1 and len(y_tr) >= 10:
                try:
                    iso, _ = calibrate_isotonic(mu_tr, y_tr, out_dir=calib_dir)
                    print(f"[Fold {f.idx}] Successfully trained isotonic calibrator.")
                except RuntimeError as e:
                    print(f"[Fold {f.idx}] WARNING: Isotonic calibration failed ({e}). Proceeding without calibrator for this fold.")
                    iso = None
            else:
                print(f"[Fold {f.idx}] WARNING: Not enough data or label variety (n={len(y_tr)}, unique_labels={len(np.unique(y_tr))}) to train calibrator. Skipping.")

            # ---- Gates from TRAIN probabilities (quantiles) ----
            p_tr = map_mu_to_prob(mu_tr, calibrator=iso) if mu_tr.size > 0 else np.array([0.5])

            # ---------- Gate selection with collapse detection ----------
            p_gate_default, full_p_default = float(self.p_gate_q), float(self.full_p_q)
            p_gate, full_p, gate_mode = p_gate_default, full_p_default, "normal"

            try:
                if isinstance(p_tr, np.ndarray) and p_tr.size:
                    p_min, p_max = float(np.nanmin(p_tr)), float(np.nanmax(p_tr))
                    p_iqr = float(np.nanpercentile(p_tr, 75) - np.nanpercentile(p_tr, 25))
                    auc_train = (float(roc_auc_score(y_tr, p_tr))
                                 if (len(np.unique(y_tr)) > 1) else float("nan"))

                    collapsed = ((p_max - p_min) < 0.05) or (p_iqr < 0.02)
                    rank_flippy = (not np.isnan(auc_train)) and (auc_train < 0.48)

                    if collapsed and rank_flippy:
                        gate_mode = "rank_flip"
                        p_tr_for_gates = 1.0 - p_tr
                        p_gate = float(np.nanpercentile(p_tr_for_gates, 50))
                        full_p = float(np.nanpercentile(p_tr_for_gates, 70))
                    elif collapsed:
                        gate_mode = "widened"
                        p_gate = float(np.nanpercentile(p_tr, 50))
                        full_p = float(np.nanpercentile(p_tr, 70))
                    else:
                        gate_mode = "normal"

                    print(f"[Gates] mode={gate_mode} p_min={p_min:.3f} p_max={p_max:.3f} "
                          f"p_IQR={p_iqr:.3f} auc_train={auc_train if not np.isnan(auc_train) else 'nan'} "
                          f"→ p_gate={p_gate:.3f} full_p={full_p:.3f}")
                else:
                    gate_mode = "no_train_p"
            except Exception as _e:
                print(f"[Gates] collapse detection failed: {type(_e).__name__}: {_e}")
                gate_mode = "error"

            '''p_gate_q = self.p_gate_q if p_gate_q is None else float(p_gate_q)
            full_p_q = self.full_p_q if full_p_q is None else float(full_p_q)
            #p_gate = float(np.quantile(p_tr, p_gate_q))
            #full_p = float(np.quantile(p_tr, full_p_q))
            # in walkforward.py where p_gate/full_p are computed from p_tr
            p_gate_raw = float(np.quantile(p_tr, p_gate_q))
            full_p_raw = float(np.quantile(p_tr, full_p_q))
            p_gate = max(p_gate_raw, 0.50)  # never go long if p<0.5
            full_p = max(full_p_raw, max(p_gate + 0.05, 0.65))  # keep separation
            '''

            # ---------- Gate selection with collapse detection ----------
            p_gate_default, full_p_default = float(self.p_gate_q), float(self.full_p_q)
            p_gate, full_p, gate_mode = p_gate_default, full_p_default, "normal"

            try:
                if 'p_tr' in locals() and isinstance(p_tr, np.ndarray) and p_tr.size:
                    p_min, p_max = float(np.nanmin(p_tr)), float(np.nanmax(p_tr))
                    p_iqr = float(np.nanpercentile(p_tr, 75) - np.nanpercentile(p_tr, 25))
                    auc_train = (float(roc_auc_score(y_tr, p_tr))
                                 if (len(np.unique(y_tr)) > 1) else float("nan"))

                    collapsed = (p_max - p_min) < 0.05 or p_iqr < 0.02
                    rank_flippy = (not np.isnan(auc_train)) and (auc_train < 0.48)

                    if collapsed and rank_flippy:
                        # (c) temporary “rank-flip” mode (diagnostic)
                        gate_mode = "rank_flip"
                        # flip probabilities for gates only
                        p_tr_for_gates = 1.0 - p_tr
                        p_gate = float(np.nanpercentile(p_tr_for_gates, 50))
                        full_p = float(np.nanpercentile(p_tr_for_gates, 70))
                    elif collapsed:
                        # (a) widen gates to be less exclusive
                        gate_mode = "widened"
                        p_gate = float(np.nanpercentile(p_tr, 50))
                        full_p = float(np.nanpercentile(p_tr, 70))
                    else:
                        # normal
                        gate_mode = "normal"

                    print(f"[Gates] mode={gate_mode} p_min={p_min:.3f} p_max={p_max:.3f} "
                          f"p_IQR={p_iqr:.3f} auc_train={auc_train if not np.isnan(auc_train) else 'nan'} "
                          f"→ p_gate={p_gate:.3f} full_p={full_p:.3f}")
                else:
                    gate_mode = "no_train_p"
            except Exception as _e:
                print(f"[Gates] collapse detection failed: {type(_e).__name__}: {_e}")
                gate_mode = "error"

            #(fold_dir / "gates.json").write_text(json.dumps({"p_gate": p_gate, "full_p": full_p}, indent=2))

            (fold_dir / "gates.json").write_text(
                json.dumps({"p_gate": float(p_gate), "full_p": float(full_p), "mode": gate_mode}, indent=2)
            )

            # ---------- TRAIN metrics (AUC) for sanity ----------
            train_auc_all = None
            try:
                if mu_tr.size > 0 and len(np.unique(y_tr)) > 1:
                    p_tr = map_mu_to_prob(mu_tr, calibrator=iso)
                    train_auc_all = float(roc_auc_score(y_tr, p_tr))
            except Exception as _e:
                train_auc_all = None

            # ---------- MANIFEST ----------
            manifest = {
                "fold_idx": int(f.idx),
                "windows": {
                    "train_start": str(f.train_start),
                    "train_end": str(f.train_end),
                    "purge_bars": int(f.purge_bars),
                    "test_start": str(f.test_start),
                    "test_end": str(f.test_end),
                },
                "gates": {"p_gate": float(p_gate), "full_p": float(full_p), "mode": gate_mode},
                "prepro_dir": str(prepro_dir),
                "ev_artifacts": {**ev_meta, "source_dir": str(self.ev_artifacts_root)},
                "env": {
                    "python": sys.version.split()[0],
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "sklearn": sklearn.__version__,
                    "platform": platform.platform(),
                },
            }

            manifest["artifact_sources"] = {
                "strategy": "pooled",
                "core_dir": paths.get("core_dir"),
                "scaler": paths.get("scaler"),
                "pca": paths.get("pca"),
                "clusters": paths.get("clusters"),
                "feature_schema": paths.get("feature_schema"),
                "ann": {
                    "trend": paths.get("ann_trend"),
                    "range": paths.get("ann_range"),
                    "vol": paths.get("ann_vol"),
                    "global": paths.get("ann_global"),
                },
                "calibrator": {
                    "path": paths.get("calibrator"),
                    "scope": paths.get("calibrator_scope"),
                    "used_in_fold": bool(calibrator_for_fold is not None),
                },
                "distance_contract": {
                    "family": dist_family,
                    "params": dist_params,
                },
            }

            cal_path = calib_dir / "iso_calibrator.pkl"
            manifest["calibrator_path"] = str(cal_path) if cal_path.exists() else None
            manifest["feature_cols_path"] = str(prepro_dir / "feature_cols.json")

            (fold_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

            # ---- TEST evaluation on SCANNED test rows ----
            #ev_te = EVEngine.from_artifacts(self.ev_artifacts_root, calibrator=iso)
            #cost_model = None if self.debug_no_costs else BasicCostModel()
            #ev_te = EVEngine.from_artifacts(self.ev_artifacts_root, calibrator=iso, cost_model=cost_model)
            ev_te = EVEngine.from_artifacts(self.ev_artifacts_root, calibrator=calibrator_for_fold,
                                            cost_model=cost_model)

            if feats_scan_te.empty:
                #(fold_dir / "decisions.parquet").write_bytes(b"")

                # Write a valid (empty) decisions file with the expected schema
                pd.DataFrame(
                    columns=["timestamp", "symbol", "mu", "p", "gate", "label"]
                ).to_parquet(fold_dir / "decisions.parquet", index=False)

                m = {
                    "fold": f.idx,
                    "train_window": [str(f.train_start), str(f.train_end - purge_cut)],
                    "test_window": [str(f.test_start), str(f.test_end)],
                    "purge_bars": f.purge_bars,
                    "auc_on_entries": float("nan"),
                    "n_test": 0,
                    "n_entries": 0,
                    "p_gate": p_gate,
                    "full_p": full_p,
                    "reliability": {"p_mean": [], "y_rate": [], "n": []},
                }
                (fold_dir / "metrics.json").write_text(json.dumps(m, indent=2))
                all_fold_metrics.append(m)
                continue

            pc_te = [c for c in feats_scan_te.columns if c.startswith("pca_")]
            X_te = feats_scan_te[pc_te].to_numpy(dtype=np.float32)
            # --- FIX: Access the .mu attribute by name ---
            #mu_te = np.array([ev_te.evaluate(x).mu for x in X_te], dtype=float)
            mu_te = _mu_on_feats(ev_te, feats_scan_te, regimes=reg_te)

            p_te = map_mu_to_prob(mu_te, calibrator=iso)

            # Labels for scanned test rows from FULL stream
            y_te = y_by_ts.reindex(scan_te["timestamp"].values).fillna(0).astype(int).to_numpy()



            #gates_passed = (p_te >= p_gate)
            # --- NEW: Relative Gating Logic ---
            # NOTE: The spec assumes a 'self.config' dictionary. Since your class doesn't have one,
            # we'll define the gate configuration here for now. You could move this to your __init__.
            gate_cfg = {"mode": "topN_per_day", "N": 10}

            # The spec uses 'p_use'. Your 'p_te' is the correct variable, as it's already calibrated.
            p_use = p_te
            dates = pd.to_datetime(scan_te["timestamp"]).dt.date.to_numpy()

            gates_passed = np.zeros_like(p_use, dtype=bool)

            if gate_cfg["mode"] == "topN_per_day":
                N = int(gate_cfg.get("N", 10))
                df_tmp = pd.DataFrame({"d": dates, "p": p_use, "original_index": np.arange(len(p_use))})
                top_indices = df_tmp.groupby("d")["p"].nlargest(N).index.get_level_values(1)
                gates_passed[df_tmp.loc[top_indices, "original_index"].values] = True

            elif gate_cfg["mode"] == "quantile_per_day":
                q = float(gate_cfg.get("quantile", 0.7))
                df_tmp = pd.DataFrame({"d": dates, "p": p_use})
                # compute daily thresholds, then mark entries
                thr = df_tmp.groupby("d")["p"].transform(lambda s: s.quantile(q)).to_numpy()
                gates_passed = p_use >= thr
            else:
                # absolute threshold (your old method)
                thr = float(gate_cfg.get("p_min", 0.55))
                gates_passed = p_use >= thr

            # (Optional) minimum total entries guard
            min_entries = 10  # Example value
            if gates_passed.sum() < min_entries:
                print(f"[Gate] WARN: entries={gates_passed.sum()} < {min_entries}; gating too strict for this fold")

            # AUC on all scanned test rows (pre-gate)
            try:
                auc_all = float(roc_auc_score(y_te, p_te)) if len(np.unique(y_te)) > 1 else float("nan")
            except ValueError:
                auc_all = float("nan")

            # AUC on entries only (post-gate, may be NaN on single class)
            try:
                mask = gates_passed
                auc_entries = float(roc_auc_score(y_te[mask], p_te[mask])) if mask.any() and len(
                    np.unique(y_te[mask])) > 1 else float("nan")
            except ValueError:
                auc_entries = float("nan")



            if train_auc_all is not None and auc_all is not None:
                if auc_all < 0.52:
                    print(f"[Fold {f.idx}] WARNING: test AUC(all)={auc_all:.3f} < 0.52")
                if train_auc_all > 0.70 and auc_all is not None and (train_auc_all - auc_all) > 0.15:
                    print(
                        f"[Fold {f.idx}] WARNING: train AUC ({train_auc_all:.3f}) >> test AUC ({auc_all:.3f}) – possible leakage or overfit")

            # ---------- Plots (reliability & ROC) ----------
            plots_dir = fold_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            def _safe_plot_reliability(y, p, out_png: Path, bins=10):
                try:
                    import matplotlib.pyplot as plt
                    if len(y) == 0 or len(np.unique(y)) < 2:
                        out_png.write_text("insufficient data")
                        return
                    # Bin by predicted prob
                    q = np.linspace(0, 1, bins + 1)
                    cuts = np.quantile(p, q)
                    idx = np.digitize(p, cuts[1:-1], right=True)
                    bin_p = []
                    bin_emp = []
                    bin_n = []
                    for b in range(bins):
                        m = idx == b
                        if m.sum() == 0:
                            continue
                        bin_p.append(p[m].mean())
                        bin_emp.append(y[m].mean())
                        bin_n.append(int(m.sum()))
                    plt.figure()
                    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
                    plt.scatter(bin_p, bin_emp, s=np.clip(np.array(bin_n), 10, 60))
                    plt.xlabel("Predicted p(up)")
                    plt.ylabel("Empirical win-rate")
                    plt.title("Reliability")
                    plt.tight_layout()
                    plt.savefig(out_png)
                    plt.close()
                except Exception as _e:
                    out_png.write_text(f"plot failed: {type(_e).__name__}: {_e}")

            def _safe_plot_roc(y, p, out_png: Path):
                try:
                    import matplotlib.pyplot as plt
                    from sklearn.metrics import roc_curve, auc
                    if len(y) == 0 or len(np.unique(y)) < 2:
                        out_png.write_text("insufficient data")
                        return
                    fpr, tpr, _ = roc_curve(y, p)
                    a = auc(fpr, tpr)
                    plt.figure()
                    plt.plot(fpr, tpr, label=f"AUC={a:.3f}")
                    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
                    plt.xlabel("FPR")
                    plt.ylabel("TPR")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(out_png)
                    plt.close()
                except Exception as _e:
                    out_png.write_text(f"plot failed: {type(_e).__name__}: {_e}")

            #_safe_plot_reliability(y_te, p_cal_te, plots_dir / "reliability_test.png")
            #_safe_plot_roc(y_te, p_cal_te, plots_dir / "roc_test.png")
            _safe_plot_reliability(y_te, p_te, plots_dir / "reliability_test.png")
            _safe_plot_roc(y_te, p_te, plots_dir / "roc_test.png")

            # Metrics
            total_rows += int(len(scan_te))
            total_entries += int(gates_passed.sum())
            try:
                auc = float(roc_auc_score(y_te[gates_passed], p_te[gates_passed])) if gates_passed.any() else float("nan")
            except ValueError:
                auc = float("nan")

            # Small-n reliability: quartiles or fewer if not enough unique p
            '''df_eval = pd.DataFrame({"p": p_te, "y": y_te})
            try:
                q = min(4, max(1, df_eval["p"].nunique()))
                df_eval["bin"] = pd.qcut(df_eval["p"], q=q, duplicates="drop")
                rel = (
                    df_eval.groupby("bin", observed=True)
                    .agg(p_mean=("p", "mean"), y_rate=("y", "mean"), n=("p", "size"))
                    .reset_index()
                )
            except Exception:
                rel = pd.DataFrame(columns=["p_mean", "y_rate", "n"])
            '''

            # Small-n reliability: quartiles or fewer if not enough unique p
            df_eval = pd.DataFrame({"p": p_te, "y": y_te})
            try:
                q = min(4, max(1, df_eval["p"].nunique() - 1))
                df_eval["bin"] = pd.qcut(df_eval["p"], q=q, duplicates="drop")
                rel = (
                    df_eval.groupby("bin", observed=True)
                    .agg(p_mean=("p", "mean"), y_rate=("y", "mean"), n=("p", "size"))
                    .reset_index()
                )
                # --- FIX: Convert the Interval objects in the 'bin' column to strings ---
                rel['bin'] = rel['bin'].astype(str)
            except Exception:
                rel = pd.DataFrame(columns=["bin", "p_mean", "y_rate", "n"])

            # Persist decisions
            out_dec = pd.DataFrame(
                {
                    "timestamp": scan_te["timestamp"].values,
                    "symbol": scan_te["symbol"].values if "symbol" in scan_te.columns else self.symbol,
                    "mu": mu_te,
                    "p": p_te,
                    "gate": gates_passed.astype(int),
                    "label": y_te,
                }
            )

            out_dec["decision_ts"] = out_dec["timestamp"]
            out_dec["h"] = int(self.horizon_bars)

            out_dec.to_parquet(fold_dir / "decisions.parquet", index=False)

            # === Phase-4: per-fold trades persistence (ADD THIS) ===================
            # Simple, label-consistent simulator:
            #  • decision made at time t (the scanned test bar)
            #  • enter at next bar open (t+1 open)
            #  • exit after H bars from decision (t+H open)
            # This matches the open→open H labeling intent and avoids look-ahead.

            H = int(self.horizon_bars)

            # Build forward timestamp and open-price maps on the FULL stream
            full_idxed = full.set_index("timestamp").sort_index()
            open_s = full_idxed["open"]
            ts_fwd_1 = full_idxed.index.to_series().shift(-1)  # timestamp of next bar
            ts_fwd_H = full_idxed.index.to_series().shift(-H)  # timestamp H bars ahead
            open_fwd_1 = open_s.shift(-1)  # next bar open
            open_fwd_H = open_s.shift(-H)  # open H bars ahead

            # Restrict to current fold’s SCANNED TEST timestamps
            ts_decisions = pd.to_datetime(scan_te["timestamp"]).values

            # Map decision bars → entry/exit timestamps and prices
            entry_ts = ts_fwd_1.reindex(ts_decisions).values
            exit_ts = ts_fwd_H.reindex(ts_decisions).values
            entry_px = open_fwd_1.reindex(ts_decisions).values
            exit_px = open_fwd_H.reindex(ts_decisions).values

            # Only keep rows that (a) passed the gate and (b) have all required future bars
            mask_gate = gates_passed.astype(bool)
            valid_future = (~pd.isna(entry_ts)) & (~pd.isna(exit_ts)) & (~pd.isna(entry_px)) & (~pd.isna(exit_px))
            keep = mask_gate & valid_future

            if keep.any():
                # Position sizing: 1 share (you can upgrade later)
                qty = np.ones(int(keep.sum()), dtype=float)

                # Costs (respect your debug flag); extend with your BasicCostModel if desired
                #commission = np.zeros_like(qty) if self.debug_no_costs else np.zeros_like(qty)
                #slippage = np.zeros_like(qty) if self.debug_no_costs else np.zeros_like(qty)

                # Costs (Phase 1.1): commission must be non-zero when enabled.
                commission_per_share = float(getattr(self, "commission", self.cfg.get("commission", 0.0)))
                slippage_bp = float(getattr(self, "slippage_bp", self.cfg.get("slippage_bp", 0.0)))

                if self.debug_no_costs:
                    commission = np.zeros_like(qty, dtype=float)
                    slippage = np.zeros_like(qty, dtype=float)
                else:
                    # charge commission on entry + exit (2 legs)
                    commission = 2.0 * commission_per_share * qty
                    # slippage modeled as bps of notional on entry + exit (2 legs)
                    slippage = 2.0 * (slippage_bp / 1e4) * entry_px[keep] * qty


                realized_pnl = (exit_px[keep] - entry_px[keep]) * qty - commission - slippage

                trades_df = pd.DataFrame({
                    "decision_ts": ts_decisions[keep],
                    "entry_ts": entry_ts[keep],
                    "exit_ts": exit_ts[keep],
                    "entry_price": entry_px[keep],
                    "exit_price": exit_px[keep],
                    "qty": qty,
                    "commission": commission,
                    "slippage": slippage,
                    "realized_pnl": realized_pnl,
                    "symbol": self.symbol,
                    "fold": int(f.idx),
                }).sort_values("entry_ts").reset_index(drop=True)

                trades_df.to_parquet(fold_dir / "trades.parquet", index=False)
            else:
                # Still write an empty file with the expected columns for consistency
                pd.DataFrame(columns=[
                    "decision_ts", "entry_ts", "exit_ts", "entry_price", "exit_price",
                    "qty", "commission", "slippage", "realized_pnl", "symbol", "fold"
                ]).to_parquet(fold_dir / "trades.parquet", index=False)
            # === End Phase-4 trades persistence =====================================

            # Persist metrics
            metrics = {
                "fold": f.idx,
                "train_window": [str(f.train_start), str(f.train_end - purge_cut)],
                "test_window": [str(f.test_start), str(f.test_end)],
                "purge_bars": f.purge_bars,
                #"auc_on_entries": auc,
                "auc_all": auc_all,
                "auc_on_entries": auc_entries,
                "train_auc_all": (None if train_auc_all is None else float(train_auc_all)),
                "n_test": int(len(scan_te)),
                "n_entries": int(gates_passed.sum()),
                "p_gate": p_gate,
                "full_p": full_p,
                "reliability": rel.to_dict(orient="list"),
            }
            (fold_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
            all_fold_metrics.append(metrics)

        agg = {
            "folds": all_fold_metrics,
            "total_entries": int(total_entries),
            "total_test_rows": int(total_rows),
        }
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        (self.artifacts_root / "oos_report.json").write_text(json.dumps(agg, indent=2))

        print("=== A2 Walk-Forward OOS Summary ===")
        for m in all_fold_metrics:
            print(
                f"Fold {m['fold']:02d}  test={m['test_window'][0]}→{m['test_window'][1]}  "
                f"rows={m['n_test']}  entries={m['n_entries']}  "
                f"AUC(all)={m.get('auc_all', float('nan')):.3f}  AUC(entries)={m['auc_on_entries']:.3f}"
            )

        print(f"TOTAL entries={agg['total_entries']}  rows={agg['total_test_rows']}")
        return agg





