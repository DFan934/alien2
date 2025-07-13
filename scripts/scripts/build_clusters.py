# ----------------------------------------------------------------------
# FILE: prediction_engine/scripts/build_clusters.py   (patched)
# ----------------------------------------------------------------------
from pathlib import Path
import numpy as np
import pandas as pd

from prediction_engine.path_cluster_engine import PathClusterEngine

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parents[2]          # repo root
PARQUET_DIR  = ROOT / "feature_engineering" / "feature_parquet"
WEIGHTS_DIR  = ROOT / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True, parents=True)

SYMBOL        = "RRC"          # <-- set your symbol here
RANDOM_STATE  = 42
MAX_ROWS      = 25_000         # down-sample cap
# ----------------------------------------------------------------------


def load_all_parquet(symbol: str) -> pd.DataFrame:
    """Concat every parquet slice under feature_parquet/symbol=… into one DF."""
    base = PARQUET_DIR / f"symbol={symbol}"
    parts: list[pd.DataFrame] = []
    for year_dir in sorted(base.iterdir()):
        if not year_dir.is_dir():
            continue
        for month_dir in sorted(year_dir.iterdir()):
            if not month_dir.is_dir():
                continue
            for pq in month_dir.glob("*.parquet"):
                parts.append(pd.read_parquet(pq))
    if not parts:
        raise FileNotFoundError(f"No parquet under {base}")
    df = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    return df


def main() -> None:
    print(f"• Loading parquet for {SYMBOL} …")
    df = load_all_parquet(SYMBOL)

    # ------------------------------------------------------------------
    # Guard: need 'close' to compute y; need pca_* for X
    # ------------------------------------------------------------------
    if "close" not in df.columns:
        raise RuntimeError("Parquet slices do not contain 'close' – cannot compute future_return.")
    pca_cols = [c for c in df.columns if c.startswith("pca_")]
    if not pca_cols:
        raise RuntimeError("No pca_* columns found – run CoreFeaturePipeline first.")

    # ------------------------------------------------------------------
    # 1) Compute forward return (log-return to next bar)
    # ------------------------------------------------------------------
    df["close_next"]    = df["close"].shift(-1)
    df["future_return"] = np.log(df["close_next"] / df["close"])

    # ------------------------------------------------------------------
    # 2) Prepare X (PCA features) and y (future_return); drop NaNs
    # ------------------------------------------------------------------
    cols_needed = pca_cols + ["future_return"]
    df_clean    = df[cols_needed].dropna()
    X           = df_clean[pca_cols].to_numpy(dtype=np.float32)
    y           = df_clean["future_return"].to_numpy(dtype=np.float32)

    print(f"  → {len(X):,} samples, {len(pca_cols)} PCA features")

    # Optional down-sample for speed
    if len(X) > MAX_ROWS:
        rs  = np.random.RandomState(RANDOM_STATE)
        idx = rs.choice(len(X), MAX_ROWS, replace=False)
        X, y = X[idx], y[idx]
        print(f"  → down-sampled to {len(X):,} rows")

    # ------------------------------------------------------------------
    # 3) Build clusters & save artefacts
    # ------------------------------------------------------------------
    k = max(3, min(15, len(X) // 2))      # simple heuristic
    print(f"• Building {k} clusters into {WEIGHTS_DIR} …")

    PathClusterEngine.build(
        X=X,
        y_numeric=y,
        y_categorical=None,        # no categorical labels yet
        feature_names=pca_cols,    # *** order matters ***
        n_clusters=k,
        out_dir=WEIGHTS_DIR,
        random_state=RANDOM_STATE,
    )
    print(f"✓ Cluster artefacts written to {WEIGHTS_DIR} (k={k})")


if __name__ == "__main__":
    main()
