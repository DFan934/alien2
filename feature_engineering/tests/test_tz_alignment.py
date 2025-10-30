import numpy as np
import pandas as pd
from feature_engineering.pipelines.core import CoreFeaturePipeline

def _toy(n=60, sym="RRC", aware: bool = True):
    ts = pd.date_range("2020-01-01 09:30", periods=n, freq="T", tz="UTC")
    if not aware:
        ts = ts.tz_convert(None)  # make tz-naive
    df = pd.DataFrame({
        "timestamp": ts,
        "symbol": sym,
        "open": np.linspace(100, 101, n, dtype=np.float32),
        "high": np.linspace(100.1, 101.1, n, dtype=np.float32),
        "low":  np.linspace( 99.9, 100.9, n, dtype=np.float32),
        "close":np.linspace(100, 101, n, dtype=np.float32),
        "volume": np.full(n, 1000, dtype=np.int32),
    })
    return df

def test_run_mem_tz_alignment_naive_vs_aware(tmp_path):
    pipe = CoreFeaturePipeline(parquet_root=tmp_path)

    # Case 1: timestamp aware, trigger_ts missing (created from timestamp)
    df_aware = _toy(aware=True)
    df_feat1, _ = pipe.run_mem(df_aware, normalization_mode="per_symbol")
    assert len(df_feat1) == len(df_aware)

    # Case 2: timestamp naive, trigger_ts missing (will be created then aligned)
    df_naive = _toy(aware=False)
    df_feat2, _ = pipe.run_mem(df_naive, normalization_mode="per_symbol")
    assert len(df_feat2) == len(df_naive)

def test_transform_mem_tz_alignment(tmp_path):
    pipe = CoreFeaturePipeline(parquet_root=tmp_path)
    df_fit = _toy(aware=True, sym="BBY")
    pipe.fit_mem(df_fit, normalization_mode="global")

    # Transform with naive timestamps
    df_naive = df_fit.copy()
    df_naive["timestamp"] = df_naive["timestamp"].dt.tz_convert(None)
    out = pipe.transform_mem(df_naive)  # should not raise
    assert out is not None
