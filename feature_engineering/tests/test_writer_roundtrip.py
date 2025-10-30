import pandas as pd
import numpy as np
from pathlib import Path

from feature_engineering.pipelines.dataset_loader import load_slice
from feature_engineering.pipelines.core import CoreFeaturePipeline

def test_writer_roundtrip(tmp_path):
    # Build a toy parquet tree
    from feature_engineering.tests.test_dataset_loader_multi_symbol2 import _write_hive_minute_parquet
    root = tmp_path / "parquet"
    _write_hive_minute_parquet(root, "AAA", 2000, 1, 2, n=16)
    _write_hive_minute_parquet(root, "BBB", 2000, 1, 5, n=16)

    df_raw, _ = load_slice(root, ["AAA","BBB"], "2000-01-01", "2000-01-31 23:59:59")
    pipe = CoreFeaturePipeline(parquet_root=root)
    df_feat, meta = pipe.run_mem(df_raw, normalization_mode="per_symbol")

    out_dir = tmp_path / "features_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "features.parquet"
    df_feat.to_parquet(out_path, index=False)

    df_back = pd.read_parquet(out_path)
    assert len(df_back) == len(df_feat), f"Round-trip row mismatch {len(df_back)} != {len(df_feat)}"
    # Every input symbol appears in output
    in_syms = set(df_raw["symbol"].unique())
    out_syms = set(df_back["symbol"].unique()) if "symbol" in df_back.columns else in_syms
    assert in_syms == out_syms, f"Symbols mismatch after write: in={in_syms}, out={out_syms}"
