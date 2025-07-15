# ----------------------------------------------------------------------
# FILE: scripts/build_clusters.py   (Corrected & Patched)
# ----------------------------------------------------------------------
from pathlib import Path
import numpy as np
import pandas as pd

from prediction_engine.path_cluster_engine import PathClusterEngine
from feature_engineering.pipelines.core import CoreFeaturePipeline
from prediction_engine.market_regime import label_days, RegimeParams

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
# Resolve paths relative to the project root
ROOT = Path(__file__).resolve().parents[1]
# --- FIX: Point to the ROOT of the parquet data, not a subfolder ---
RAW_PARQUET_DIR = ROOT / "parquet"
WEIGHTS_DIR = ROOT / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True, parents=True)

SYMBOL = "RRC"
RANDOM_STATE = 42
MAX_ROWS = 25_000
# ----------------------------------------------------------------------

# ------------------------------------------------------------------
# Replace load_raw_data in scripts/build_clusters.py
# ------------------------------------------------------------------
def load_raw_data(symbol: str) -> pd.DataFrame:
    """Load all raw minute bars for one symbol from partitioned Parquet."""
    # Point directly at the symbol partition to avoid schema.json
    symbol_path = RAW_PARQUET_DIR / f"symbol={symbol}"
    if not symbol_path.exists():
        raise FileNotFoundError(f"No Parquet data found at {symbol_path}")

    df = pd.read_parquet(symbol_path)          # recurse into year=/month=/
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df



def main() -> None:
    print(f"• Loading RAW minute-bar data for {SYMBOL}…")
    raw_df = load_raw_data(SYMBOL)

    # ------------------------------------------------------------------
    # 1. Run the full feature engineering pipeline
    # ------------------------------------------------------------------
    print("• Engineering features (including PCA)...")
    pipe = CoreFeaturePipeline(parquet_root=Path(""))  # In-memory
    feats_df, _ = pipe.run_mem(raw_df)
    pca_cols = [c for c in feats_df.columns if c.startswith("pca_")]
    if not pca_cols:
        raise RuntimeError("No pca_* columns found – did CoreFeaturePipeline run?")
    print(f"  → {len(pca_cols)} PCA features generated.")

    # ------------------------------------------------------------------
    # 2. Generate Regime Labels from Daily Resampled Data
    # ------------------------------------------------------------------
    # 2. Generate daily regimes (unchanged)
    daily_df = (
        raw_df
        .set_index('timestamp')
        .resample('D')
        .agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
        .dropna()
    )
    daily_regimes = label_days(daily_df, RegimeParams())

    # ─── Convert the regime‐index to plain, naive Timestamps ───
    if daily_regimes.index.tz is not None:
        daily_regimes.index = daily_regimes.index.tz_localize(None)

    # ─── Build a helper “date” column in feats_df ───
    # (this avoids any in‐place timezone gymnastics on your real timestamp)
    #feats_df['date'] = pd.to_datetime(feats_df['timestamp']).dt.normalize()

    # ─── Drop any UTC tzinfo on timestamp so mapping to naive daily index works ───
    if feats_df['timestamp'].dt.tz is not None:
        feats_df['timestamp'] = feats_df['timestamp'].dt.tz_localize(None)
    # ─── Build a helper “date” column in feats_df ───
    # (normalize to midnight, matching daily_regimes.index)
    feats_df['date'] = feats_df['timestamp'].dt.normalize()


    # ─── Now map from that date to the regime Series ───
    feats_df['regime'] = feats_df['date'].map(daily_regimes)

    # ─── Fill through any holes ───
    feats_df['regime'].ffill(inplace=True)
    feats_df['regime'].bfill(inplace=True)

    print(f"  → {feats_df['regime'].isna().sum()} missing regimes (should be 0)")

    # ─── (Optionally) drop the helper column when you’re done ───
    feats_df.drop(columns=['date'], inplace=True)

    print("  → Regime labels generated and mapped.")

    # ------------------------------------------------------------------
    # 3. Compute forward return and prepare final data for clustering
    # ------------------------------------------------------------------

    # Ensure both timestamp columns are timezone‐naive, so merge will work
    feats_df['timestamp'] = feats_df['timestamp'].dt.tz_localize(None)
    raw_df['timestamp'] = raw_df['timestamp'].dt.tz_localize(None)

    # The feature pipeline might drop the 'close' column, so we merge it back
    if 'close' not in feats_df.columns:
        feats_df = feats_df.merge(raw_df[['timestamp', 'close']], on='timestamp', how='left')

    feats_df["future_return"] = np.log(feats_df["close"].shift(-1) / feats_df["close"])

    cols_needed = pca_cols + ["future_return", "regime"]
    df_clean = feats_df[cols_needed].dropna()

    X = df_clean[pca_cols].to_numpy(dtype=np.float32)
    y_numeric = df_clean["future_return"].to_numpy(dtype=np.float32)
    y_categorical = df_clean["regime"]

    print(f"  → Final dataset size for clustering: {len(X)} samples.")

    # Optional down-sample for speed
    if len(X) > MAX_ROWS:
        print(f"  → Down-sampling to {MAX_ROWS} rows...")
        rs = np.random.RandomState(RANDOM_STATE)
        idx = rs.choice(len(X), MAX_ROWS, replace=False)
        X, y_numeric, y_categorical = X[idx], y_numeric[idx], y_categorical.iloc[idx]

    # ------------------------------------------------------------------
    # 4. Build clusters & save artefacts
    # ------------------------------------------------------------------
    k = max(3, min(15, len(X) // 2))
    print(f"• Building {k} clusters into {WEIGHTS_DIR}…")

    PathClusterEngine.build(
        X=X,
        y_numeric=y_numeric,
        y_categorical=y_categorical,
        feature_names=pca_cols,
        n_clusters=k,
        out_dir=WEIGHTS_DIR,
        random_state=RANDOM_STATE,
    )
    print(f"✓ Cluster artefacts written to {WEIGHTS_DIR} (k={k})")


if __name__ == "__main__":
    main()
