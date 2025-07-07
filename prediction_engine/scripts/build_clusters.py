from pathlib import Path
import numpy as np
import pandas as pd
from prediction_engine.path_cluster_engine import PathClusterEngine

# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]           # repo root
PARQUET_DIR = ROOT / "feature_engineering" / "feature_parquet"
WEIGHTS_DIR  = ROOT / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)
# ----------------------------------------------------------------------

def biggest_parquet() -> Path:
    files = list(PARQUET_DIR.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No Parquet under {PARQUET_DIR}. Run feature_engineering/run_pipeline.py first."
        )
    return max(files, key=lambda p: p.stat().st_size)


def main() -> None:
    pq_path = biggest_parquet()
    print(f"• Loading {pq_path.relative_to(ROOT)}")

    df = pd.read_parquet(pq_path)

    # ---- keep only numeric *feature* columns -------------------------
    NUM = (
        df.select_dtypes("number")
          .drop(columns=["open", "high", "low", "close", "volume"],
                errors="ignore")
    )
    feature_names = list(NUM.columns)          # should be six names
    NUM = NUM.dropna()  # <— drop rows with any NaN
    X = NUM.to_numpy(np.float32)

    # ---- outcome proxy (replace with true future_return if you have it)
    if "future_return" in df:                  # ideal case
        y = df["future_return"].to_numpy(np.float32)
    else:                                      # placeholder: first numeric col
        y = NUM.iloc[:, 0].to_numpy(np.float32)

    # ---- down-sample to keep k-means fast (25k rows is plenty) --------
    if len(X) > 25_000:
        rs = np.random.RandomState(42)
        idx = rs.choice(len(X), 25_000, replace=False)
        X, y = X[idx], y[idx]

    k = max(3, min(15, len(X) // 2))           # heuristic cluster count
    PathClusterEngine.build(
        X,
        y,
        feature_names=feature_names,
        n_clusters=k,
        out_dir=WEIGHTS_DIR,
        random_state=42,
    )
    print(f"✓ wrote weights/ (k={k}, features={feature_names})")

if __name__ == "__main__":
    main()
