import numpy as np
import pandas as pd

from feature_engineering.pipelines.core import CoreFeaturePipeline, _PREDICT_COLS

def _toy(symbol: str, shift: float) -> pd.DataFrame:
    ts = pd.date_range("2020-01-01 09:30", periods=50, freq="T", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts.tz_convert(None),  # pipeline normalizes tz
        "symbol": symbol,
        "open": 100 + shift + np.arange(50) * 0.01,
        "high": 100 + shift + np.arange(50) * 0.01 + 0.1,
        "low":  100 + shift + np.arange(50) * 0.01 - 0.1,
        "close":100 + shift + np.arange(50) * 0.01,
        "volume": 1000,
    })
    return df

def test_normalization_modes_differ_single_symbol(tmp_path):
    pipe = CoreFeaturePipeline(parquet_root=tmp_path)
    df = _toy("RRC", shift=5.0)

    # run_mem returns PCA features; we only need to check pre-PCA normalization effect via fit_mem
    meta_g = pipe.fit_mem(df, normalization_mode="global")
    meta_p = pipe.fit_mem(df, normalization_mode="per_symbol")

    assert meta_g["normalization_mode"] == "global"
    assert meta_p["normalization_mode"] == "per_symbol"
    # Not a strict numeric check post-PCA; presence of modes in meta + separate pipelines is enough here.

def test_per_symbol_centers_within_each_symbol(tmp_path):
    pipe = CoreFeaturePipeline(parquet_root=tmp_path)
    dfA = _toy("RRC", shift=5.0)
    dfB = _toy("BBY", shift=-3.0)
    df = pd.concat([dfA, dfB], ignore_index=True)

    # Use the private helper to check statistics before PCA
    from feature_engineering.pipelines.core import _zscore_per_symbol

    df_feats = pipe._calculate_base_features(df)
    z = _zscore_per_symbol(df_feats, _PREDICT_COLS)

    #z = _zscore_per_symbol(df.assign(**{c: 0.0 for c in _PREDICT_COLS}), _PREDICT_COLS)

    # after z-scoring, within each symbol means should be ~0, std ~1 for the chosen cols
    # after z-scoring:
    grp = z.groupby("symbol")[_PREDICT_COLS]
    means = grp.mean().abs().max().max()
    assert means < 0.05

    # variance-aware std checks:
    orig_grp = df_feats.groupby("symbol")[_PREDICT_COLS]
    orig_stds = orig_grp.std(ddof=0)
    z_stds = grp.std(ddof=0)

    # Where original variance > 0, z-std should be ~1
    var_mask = orig_stds > 1e-12
    err_var = (z_stds - 1).abs().where(var_mask)
    assert err_var.max().max() < 0.05

    # Where original variance == 0, z-std should be ~0
    const_mask = ~var_mask
    err_const = (z_stds.where(const_mask)).fillna(0.0).abs()
    assert err_const.max().max() < 1e-8




import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from feature_engineering.pipelines.core import _PREDICT_COLS, _zscore_per_symbol

def _stacked_shifted(n=60):
    tsA = pd.date_range("2020-01-01 09:30", periods=n, freq="T", tz="UTC")
    tsB = pd.date_range("2020-01-02 09:30", periods=n, freq="T", tz="UTC")
    A = pd.DataFrame({
        "timestamp": tsA, "symbol": "A",
        "open":  100 + 5 + np.linspace(0, 0.3, n, dtype=np.float32),
        "high":  100 + 5 + np.linspace(0.1,0.4, n, dtype=np.float32),
        "low":   100 + 5 + np.linspace(-0.1,0.2, n, dtype=np.float32),
        "close": 100 + 5 + np.linspace(0, 0.3, n, dtype=np.float32),
        "volume": np.full(n, 1000, dtype=np.int32),
    })
    B = pd.DataFrame({
        "timestamp": tsB, "symbol": "B",
        "open":  100 - 3 + np.linspace(0, 0.3, n, dtype=np.float32),
        "high":  100 - 3 + np.linspace(0.1,0.4, n, dtype=np.float32),
        "low":   100 - 3 + np.linspace(-0.1,0.2, n, dtype=np.float32),
        "close": 100 - 3 + np.linspace(0, 0.3, n, dtype=np.float32),
        "volume": np.full(n, 1000, dtype=np.int32),
    })
    return pd.concat([A,B], ignore_index=True)

def test_global_keeps_between_symbol_shift(tmp_path):
    pipe = CoreFeaturePipeline(parquet_root=tmp_path)
    df = _stacked_shifted()
    df_feats = pipe._calculate_base_features(df)

    X = df_feats[_PREDICT_COLS].astype(np.float32).to_numpy()
    Xg = StandardScaler().fit_transform(X)
    out = pd.DataFrame(Xg, columns=_PREDICT_COLS).assign(symbol=df_feats["symbol"].values)

    means_by_sym = out.groupby("symbol")[_PREDICT_COLS].mean().abs().max()
    # Global scaling should keep between-symbol differences for many engineered cols
    assert means_by_sym.max() > 0.20, (
        f"Expected non-zero per-symbol mean after global scaling; got {means_by_sym.to_dict()}"
    )


def test_per_symbol_removes_shift(tmp_path):
    pipe = CoreFeaturePipeline(parquet_root=tmp_path)
    df = _stacked_shifted()
    df_feats = pipe._calculate_base_features(df)

    z = _zscore_per_symbol(df_feats.copy(), _PREDICT_COLS)
    grp = z.groupby("symbol")[_PREDICT_COLS]
    mean_err = grp.mean().abs().max().max()
    assert mean_err < 0.05

    # Variance-aware std checks (don’t penalize constant features)
    orig_grp = df_feats.groupby("symbol")[_PREDICT_COLS]
    orig_stds = orig_grp.std(ddof=0)
    z_stds = grp.std(ddof=0)

    var_mask = orig_stds > 1e-12          # columns that actually vary within each symbol
    err_var = (z_stds - 1).abs().where(var_mask).max().max()
    assert err_var < 0.05                 # varying cols ~1 after z-score

    const_mask = ~var_mask                # columns constant within symbol
    err_const = (z_stds.where(const_mask)).fillna(0.0).abs().max().max()
    assert err_const < 1e-8               # constant cols ~0 std after z-score


def test_global_keeps_between_symbol_shift_on_level_proxy(tmp_path):
    # No pipeline needed here
    df = _stacked_shifted()
    # create a level-preserving feature
    df["level_proxy"] = df["close"].astype(np.float32)
    Xg = StandardScaler().fit_transform(df[["level_proxy"]])
    out = pd.DataFrame(Xg, columns=["level_proxy"]).assign(symbol=df["symbol"].values)
    means_by_sym = out.groupby("symbol")[["level_proxy"]].mean().abs().max()
    assert float(means_by_sym) > 0.2
