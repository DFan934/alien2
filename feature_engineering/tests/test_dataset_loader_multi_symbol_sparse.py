

# feature_engineering/tests/test_dataset_loader_multi_symbol_sparse.py
import numpy as np
import pandas as pd
from feature_engineering.pipelines.core import CoreFeaturePipeline

def _toy(sym, start="2020-01-01 09:30", n=60):
    ts = pd.date_range(start, periods=n, freq="T", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts.tz_convert(None),  # pipeline normalizes tz
        "symbol": sym,
        "open": 100.0,
        "high": 100.2,
        "low": 99.8,
        "close": 100.1,
        "volume": 1000,
    })

def test_sparse_coverage_two_symbols_run_mem(tmp_path):
    # A has 60 minutes; B has only first 10 minutes → sparse per-symbol coverage
    dfA = _toy("RRC", n=60)
    dfB = _toy("BBY", n=10)
    df  = pd.concat([dfA, dfB], ignore_index=True)

    pipe = CoreFeaturePipeline(parquet_root=tmp_path)
    out, meta = pipe.run_mem(df, normalization_mode="per_symbol")

    # Qual: did not crash; Quant: produced rows for both symbols present in input
    assert set(out["symbol"].unique()) == {"RRC", "BBY"}
    # Ensure we produced at least the available rows for BBY (exact count depends on calculators’ warmups)
    assert (out[out["symbol"] == "BBY"].shape[0]) > 0
    assert "pca_1" in out.columns  # PCA applied
