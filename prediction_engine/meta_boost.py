# prediction_engine/meta_boost.py — tiny GBM meta‑learner
# ----------------------------------------------------------------------
"""Train a lightweight GradientBoostingRegressor that maps
    [mu_syn, mu_kernel, var_syn, var_kernel] → realised_return

The model is serialized to **artifacts/gbm_meta.pkl** and loaded at runtime by
EVEngine for a convex blend with the regular estimate.

* No CLI: simply press ▶ in your IDE. If default Parquet files aren’t present
  it falls back to a synthetic self‑test.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import warnings
import joblib

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

DEFAULT_FEATURES_PATH = Path("store/features.parquet")  # must include helper cols
DEFAULT_LABELS_PATH = Path("store/labels.parquet")      # realised returns
ARTIFACT_DIR = Path("prediction_engine/artifacts")
MODEL_FILE = ARTIFACT_DIR / "gbm_meta.pkl"


@dataclass(slots=True)
class MetaInfo:
    date: str
    n_rows: int
    rmse_val: float
    params: dict


def _load(parsed: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    if DEFAULT_FEATURES_PATH.exists() and DEFAULT_LABELS_PATH.exists():
        X = pd.read_parquet(DEFAULT_FEATURES_PATH)
        y = pd.read_parquet(DEFAULT_LABELS_PATH).iloc[:, -1]
        return X, y
    return _synthetic()


def _synthetic() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = 4000
    mu_syn = rng.normal(0, 0.02, n)
    mu_k   = mu_syn + rng.normal(0, 0.01, n)
    var_s  = rng.uniform(0.0001, 0.0004, n)
    var_k  = var_s + rng.uniform(-0.00005, 0.00005, n)
    X = pd.DataFrame({
        "mu_syn": mu_syn,
        "mu_kernel": mu_k,
        "var_syn": var_s,
        "var_kernel": var_k,
    })
    y = mu_syn + rng.normal(0, 0.02, n)
    return X, pd.Series(y, name="ret")


def train_and_save() -> None:
    X, y = _load()
    if {"mu_syn", "mu_kernel", "var_syn", "var_kernel"} - set(X.columns):
        warnings.warn("Meta-features missing; skipping GBM train")
        return

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    gbm = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.7,
        random_state=42,
    )
    gbm.fit(X_train, y_train)
    # mean_squared_error(squared=…) not available in older sklearn.
    mse  = mean_squared_error(y_val, gbm.predict(X_val))
    rmse = float(mse ** 0.5)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(gbm, MODEL_FILE)

    meta = MetaInfo(
        date=datetime.utcnow().isoformat(timespec="seconds"),
        n_rows=len(X),
        rmse_val=float(rmse),
        params={k: getattr(gbm, k) for k in ("n_estimators", "learning_rate", "subsample")},
    )
    import json
    (ARTIFACT_DIR / "gbm_meta.json").write_text(json.dumps(asdict(meta), indent=2))
    print(f"[meta_boost] model saved  val_RMSE={rmse:.5f}")


if __name__ == "__main__":
    train_and_save()
