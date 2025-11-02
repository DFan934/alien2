# feature_engineering/tests/test_dataset_loader_multi_symbol_no_lookahead.py
import pandas as pd
from feature_engineering.pipelines.core import CoreFeaturePipeline

def _toy(sym, start="2020-01-01 09:30", n=5):
    ts = pd.date_range(start, periods=n, freq="T", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts.tz_convert(None),
        "symbol": sym,
        "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1
    })

def test_no_lookahead_per_symbol(tmp_path):
    df = pd.concat([_toy("RRC", n=5), _toy("BBY", n=3)], ignore_index=True)
    pipe = CoreFeaturePipeline(parquet_root=tmp_path)
    feats, _ = pipe.run_mem(df, normalization_mode="global")
    # timestamps must be non-decreasing per symbol
    for sym in feats["symbol"].unique():
        ts = feats.loc[feats["symbol"] == sym, "timestamp"].values
        assert all(ts[i] <= ts[i+1] for i in range(len(ts)-1))
