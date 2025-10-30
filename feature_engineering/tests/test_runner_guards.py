import numpy as np
import pandas as pd
import pytest

from feature_engineering.pipelines.core import CoreFeaturePipeline, _PREDICT_COLS

def _toy_df(n=100, sym="RRC", start="2020-01-01 09:30"):
    ts = pd.date_range(start, periods=n, freq="T", tz="UTC")
    df = pd.DataFrame({
        "timestamp": ts,
        "symbol": sym,
        "open": np.linspace(100.0, 101.0, n, dtype=np.float32),
        "high": np.linspace(100.1, 101.1, n, dtype=np.float32),
        "low":  np.linspace( 99.9, 100.9, n, dtype=np.float32),
        "close":np.linspace(100.0, 101.0, n, dtype=np.float32),
        "volume": np.full(n, 1000, dtype=np.int32),
    })
    return df

def test_run_mem_accepts_missing_trigger_ts(tmp_path):
    pipe = CoreFeaturePipeline(parquet_root=tmp_path)
    df_raw = _toy_df(n=120, sym="RRC")

    # Intentionally no 'trigger_ts'
    assert "trigger_ts" not in df_raw.columns

    df_feat, meta = pipe.run_mem(df_raw, normalization_mode="per_symbol")
    # Row counts preserved (± any documented warmups; here there are none)
    assert len(df_feat) == len(df_raw)
    assert meta.get("normalization_mode") == "per_symbol"

def test_transform_mem_accepts_missing_trigger_ts(tmp_path):
    pipe = CoreFeaturePipeline(parquet_root=tmp_path)
    df_raw = _toy_df(n=150, sym="BBY")

    # Fit once (creates pca_meta.json etc.)
    pipe.fit_mem(df_raw, normalization_mode="global")

    # Now transform a slice *without* trigger_ts
    df_slice = df_raw.iloc[:50].drop(columns=[], errors="ignore")  # no trigger_ts
    out = pipe.transform_mem(df_slice)
    assert out is not None  # simply asserts no exception
