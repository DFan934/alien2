# prediction_engine/testing_validation/walkforward.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from feature_engineering.pipelines.core import CoreFeaturePipeline
from prediction_engine.ev_engine import EVEngine
from prediction_engine.calibration import calibrate_isotonic, map_mu_to_prob

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
            folds.append(
                Fold(
                    idx=len(folds) + 1,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
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
    ) -> Dict:
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)

        # filter both frames to window (defensive)
        full = df_full[(df_full["timestamp"] >= start) & (df_full["timestamp"] <= end)].reset_index(drop=True)
        scan = df_scanned[(df_scanned["timestamp"] >= start) & (df_scanned["timestamp"] <= end)].reset_index(drop=True)
        if full.empty:
            raise ValueError("No rows in df_full after date filter")
        # Note: scan MAY be empty in a given month; we’ll still compute folds/metrics.

        # Precompute next-bar label on FULL stream (binary up on next bar)
        full = full.sort_values("timestamp").reset_index(drop=True)
        ret1_full = (full["close"].shift(-1) / full["open"].shift(-1) - 1.0)
        y_full = (ret1_full.fillna(0.0) > 0).astype(int).rename("y")
        # Align helper: quick index by timestamp
        y_by_ts = pd.Series(y_full.values, index=full["timestamp"].values)

        folds = self._build_folds(start, end, full)

        all_fold_metrics = []
        total_entries = 0
        total_rows = 0

        for f in folds:
            fold_dir = self.artifacts_root / f"fold_{f.idx:02d}"
            calib_dir = fold_dir / "calibration"
            plots_dir = fold_dir / "plots"
            for d in (calib_dir, plots_dir):
                d.mkdir(parents=True, exist_ok=True)

            bar_dt = self._median_bar_delta(full)
            purge_cut = f.purge_bars * bar_dt

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
            feats_scan_tr = feats_full_tr.loc[feats_full_tr['timestamp'].isin(scan_tr['timestamp'])].reset_index(
                drop=True)
            feats_scan_te = feats_full_te.loc[feats_full_te['timestamp'].isin(scan_te['timestamp'])].reset_index(
                drop=True)

            print(
                f"[Fold {f.idx}] Found {len(feats_scan_tr)} scanned train bars and {len(feats_scan_te)} scanned test bars.")

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

            #ev_tr = EVEngine.from_artifacts(self.ev_artifacts_root)
            # Decide cost handling once and pass it into both train/test engines
            cost_model = None if self.debug_no_costs else BasicCostModel()
            ev_tr = EVEngine.from_artifacts(self.ev_artifacts_root, cost_model=cost_model)
            # Snapshot the EV artefacts that were just (re)built for this fold
            ev_src = Path(self.ev_artifacts_root)
            ev_dst = fold_dir / "ev"
            ev_dst.mkdir(parents=True, exist_ok=True)
            for p in ev_src.iterdir():
                if p.is_file():
                    shutil.copy2(p, ev_dst / p.name)

            # Try to load EV meta for manifest
            ev_meta = {}
            try:
                ev_meta_path = ev_src / "meta.json"
                if ev_meta_path.exists():
                    ev_meta = json.loads(ev_meta_path.read_text())
            except Exception as _e:
                ev_meta = {"error": f"meta read failed: {type(_e).__name__}"}

            def _mu_on_feats(ev: EVEngine, feats_df: pd.DataFrame) -> np.ndarray:
                if feats_df.empty:
                    return np.array([], dtype=float)
                pc_cols = [c for c in feats_df.columns if c.startswith("pca_")]
                X = feats_df[pc_cols].to_numpy(dtype=np.float32)
                # --- FIX: Access the .mu attribute by name ---
                return np.array([ev.evaluate(x).mu for x in X], dtype=float)


            mu_tr = _mu_on_feats(ev_tr, feats_scan_tr)

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

            cal_path = calib_dir / "iso_calibrator.pkl"
            manifest["calibrator_path"] = str(cal_path) if cal_path.exists() else None
            manifest["feature_cols_path"] = str(prepro_dir / "feature_cols.json")

            (fold_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

            # ---- TEST evaluation on SCANNED test rows ----
            #ev_te = EVEngine.from_artifacts(self.ev_artifacts_root, calibrator=iso)
            #cost_model = None if self.debug_no_costs else BasicCostModel()
            ev_te = EVEngine.from_artifacts(self.ev_artifacts_root, calibrator=iso, cost_model=cost_model)

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
            mu_te = np.array([ev_te.evaluate(x).mu for x in X_te], dtype=float)
            p_te = map_mu_to_prob(mu_te, calibrator=iso)

            # Labels for scanned test rows from FULL stream
            y_te = y_by_ts.reindex(scan_te["timestamp"].values).fillna(0).astype(int).to_numpy()

            gates_passed = (p_te >= p_gate)

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
            out_dec.to_parquet(fold_dir / "decisions.parquet", index=False)

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





