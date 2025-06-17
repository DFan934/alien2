import pandas as pd
from pathlib import Path
from data_ingestion.persistence import write_parquet

def test_csv_pipeline_roundtrip(csv_pipeline, sample_csv, tmp_path):
    df = csv_pipeline.parse(sample_csv, symbol="SPY")

    # essential columns present & dtypes correct
    expected_cols = ["timestamp", "open", "high", "low", "close", "volume", "symbol"]
    assert list(df.columns) == expected_cols
    assert df["timestamp"].dtype == "datetime64[ns, UTC]"

    dest = tmp_path / "roundtrip.parquet"
    write_parquet(df, dest, partition_cols=["symbol"])
    df2 = pd.read_parquet(dest)
    df2["symbol"] = df2["symbol"].astype(object)  # <- add this line

    # identical after round-trip
    pd.testing.assert_frame_equal(df.reset_index(drop=True), df2.reset_index(drop=True))
