import pandas as pd
import numpy as np
from feature_engineering.pipelines.core import CoreFeaturePipeline

def _toy_df(n=50):
    ts = pd.date_range("2020-01-01 14:30", periods=n, freq="T", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "symbol": ["TEST"] * n,
        "open": np.linspace(10, 11, n),
        "high": np.linspace(10.1, 11.1, n),
        "low":  np.linspace(9.9, 10.9, n),
        "close": np.linspace(10, 11, n),
        "volume": np.ones(n) * 1000,
        # optional scanner cols:
        "trigger_ts": ts,
        "volume_spike_pct": np.zeros(n),
    })

def test_run_mem_output_timestamp_is_utc_aware(tmp_path):
    pipe = CoreFeaturePipeline(parquet_root=tmp_path)
    df = _toy_df()
    feats, meta = pipe.run_mem(df)
    assert "timestamp" in feats.columns
    assert str(feats["timestamp"].dtype) == "datetime64[ns, UTC]"


def test_transform_mem_output_timestamp_is_utc_aware(tmp_path):
    pipe = CoreFeaturePipeline(parquet_root=tmp_path)

    df_train = _toy_df(80)
    pipe.fit_mem(df_train)  # persists pipeline.pkl in tmp_path/_fe_meta

    df_eval = _toy_df(60)
    feats = pipe.transform_mem(df_eval)

    assert "timestamp" in feats.columns
    assert str(feats["timestamp"].dtype) == "datetime64[ns, UTC]"
