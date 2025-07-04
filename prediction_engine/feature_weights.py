from __future__ import annotations

"""Random‑Forest Feature‑Weight Trainer
======================================

Purpose
-------
Generate a **feature‑importance weight vector** (`weights.npy`) used by the
`rf_weighted` distance metric.

* Run it two ways:
  1. **Press ▶️ in your IDE** – if the default Parquet files exist the model
     trains immediately (perfect for nightly cron or manual runs).
  2. **CLI** – point at custom feature/label files:
     ```bash
     python -m prediction_engine.feature_weights \
         --features path/to/features.parquet \
         --labels   path/to/labels.parquet
     ```

Outputs
~~~~~~~
* ``artifacts/weights.npy``           – float32 importance vector (∑w = 1)
* ``artifacts/weights_meta.json``     – run metadata (date, RF params, score)

Algorithm
~~~~~~~~~
1. Load engineered features (rows = setups, cols = numeric features).
2. Load outcome labels (up / down / flat).  If the label column is numeric
   returns, auto‑bucket into three classes using ±0.5σ cut‑points.
3. Train a ``RandomForestClassifier`` (500 trees, class‑balanced).
4. Extract ``feature_importances_``, normalise to sum‑to‑one.
5. Save the weight vector + JSON meta.

The script falls back to a synthetic self‑test if neither default path nor CLI
args are supplied, ensuring the ▶️ run always exits gracefully.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import json
import sys
import warnings

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import balanced_accuracy_score
except ImportError as e:  # pragma: no cover – diagnostics
    raise RuntimeError("scikit‑learn is required for feature weight training") from e

# ---------------------------------------------------------------------
DEFAULT_FEATURES_PATH = Path("store/features.parquet")
DEFAULT_LABELS_PATH = Path("store/labels.parquet")
ARTIFACT_DIR = Path("prediction_engine/artifacts")


@dataclass(slots=True)
class RunMeta:
    date: str
    n_rows: int
    n_features: int
    rf_params: dict
    val_bal_acc: float
    feature_sha1: str


# ---------------------------------------------------------------------
# ---- helpers ---------------------------------------------------------


def _bucket_outcomes(y: pd.Series) -> pd.Series:
    """Map numeric returns → \{-1, 0, 1\} buckets around ±0.5σ."""
    if y.dtype.kind in "if":
        thresh = 0.5 * y.std(ddof=0)
        return pd.cut(y, bins=[-np.inf, -thresh, thresh, np.inf], labels=[-1, 0, 1]).astype(int)
    return y.astype(int)


def _sha1_feature_order(cols: list[str]) -> str:
    import hashlib
    h = hashlib.sha1()
    h.update("::".join(cols).encode())
    return h.hexdigest()[:12]


# ---------------------------------------------------------------------
# ---- core ------------------------------------------------------------


def train_weights(features_path: Path, labels_path: Path) -> None:
    # 1. Load data ----------------------------------------------------
    df_x = pd.read_parquet(features_path)
    df_y = pd.read_parquet(labels_path) if labels_path.suffix == ".parquet" else pd.read_csv(labels_path)

    df = df_x.join(df_y.set_index(df_y.columns[0]), how="inner")
    if df.empty:
        raise ValueError("No overlapping setup_id rows between features and labels")

    y = _bucket_outcomes(df.iloc[:, -1])  # label column is last after join
    X = df.iloc[:, :-1].astype(np.float32)

    # 2. Train/valid split ------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 3. Random‑Forest training -------------------------------------
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_val)
    bal_acc = balanced_accuracy_score(y_val, y_pred)

    # 4. Importance vector ------------------------------------------
    w = rf.feature_importances_.astype(np.float32)
    if w.sum() == 0:
        warnings.warn("All feature importances are zero; falling back to uniform weights")
        w[:] = 1.0
    w /= w.sum()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(ARTIFACT_DIR / "weights.npy", w)

    meta = RunMeta(
        date=datetime.utcnow().isoformat(timespec="seconds"),
        n_rows=len(df),
        n_features=X.shape[1],
        rf_params={k: getattr(rf, k) for k in ["n_estimators", "max_depth", "min_samples_leaf"]},
        val_bal_acc=float(bal_acc),
        feature_sha1=_sha1_feature_order(list(X.columns)),
    )
    (ARTIFACT_DIR / "weights_meta.json").write_text(json.dumps(asdict(meta), indent=2))
    print(f"[feature_weights] wrote weights.npy  val_bal_acc={bal_acc:.3f}")


# ---------------------------------------------------------------------
# ---- entrypoints -----------------------------------------------------


def _self_test():  # pragma: no cover – quick synthetic run
    print("[feature_weights] running self‑test on synthetic data …")
    n, d = 5000, 30
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(n, d)), columns=[f"f{i}" for i in range(d)])
    y = pd.Series(rng.integers(-1, 2, size=n), name="outcome")
    tmp_f = Path("/tmp/feat.parquet"); tmp_l = Path("/tmp/lab.parquet")
    X.to_parquet(tmp_f); y.to_frame().to_parquet(tmp_l)
    train_weights(tmp_f, tmp_l)
    print("[feature_weights] self‑test completed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Random‑Forest feature importance trainer")
    parser.add_argument("--features", type=Path, default=None, help="Path to features Parquet")
    parser.add_argument("--labels", type=Path, default=None, help="Path to labels Parquet/CSV")
    args = parser.parse_args()

    if args.features is None and args.labels is None:
        if DEFAULT_FEATURES_PATH.exists() and DEFAULT_LABELS_PATH.exists():
            train_weights(DEFAULT_FEATURES_PATH, DEFAULT_LABELS_PATH)
        else:
            _self_test()
    else:
        if args.features is None or args.labels is None:
            parser.error("--features and --labels must be provided together")
        train_weights(args.features, args.labels)
