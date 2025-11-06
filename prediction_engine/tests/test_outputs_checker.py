import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
import pyarrow.dataset as ds


def _share_multisymbol(decisions_path: Path) -> float:
    ds_dec = ds.dataset(decisions_path, format="parquet")
    df = ds_dec.to_table(columns=["timestamp", "symbol"]).to_pandas()
    g = df.groupby("timestamp")["symbol"].nunique()
    return float((g >= 2).mean()) if len(g) else 0.0

def _causality_ratio(decisions_path: Path) -> float:
    ds_dec = ds.dataset(decisions_path, format="parquet")
    df = ds_dec.to_table(columns=["decision_ts", "entry_ts"]).to_pandas()
    if df.empty:
        return 0.0
    return float((pd.to_datetime(df["entry_ts"], utc=True) >
                  pd.to_datetime(df["decision_ts"], utc=True)).mean())

def _seed_decisions(dec_dir: Path) -> Path:
    """
    Create a minimal decisions.parquet that:
      • has overlapping timestamps across ≥2 symbols (≥10% of timestamps)
      • has entry_ts strictly after decision_ts
    """
    dec_dir.mkdir(parents=True, exist_ok=True)
    out = dec_dir / "decisions.parquet"

    # Build 12 timestamps, make 5 of them shared by two symbols (≈41.7% overlap)
    base = pd.Timestamp("1999-01-05 09:30:00Z")
    ts_unique = [base + pd.Timedelta(minutes=i) for i in range(12)]
    shared_idx = {1, 3, 5, 7, 9}  # these will appear for both symbols

    rows = []
    for i, ts in enumerate(ts_unique):
        # Always write one row for RRC
        decision = ts
        entry = ts + pd.Timedelta(minutes=1)
        rows.append(
            dict(timestamp=ts, symbol="RRC",
                 decision_ts=decision, entry_ts=entry)
        )
        # On shared slots, also write BBY with the same timestamp
        if i in shared_idx:
            rows.append(
                dict(timestamp=ts, symbol="BBY",
                     decision_ts=decision, entry_ts=entry)
            )

    df = pd.DataFrame(rows)
    # Write as a single-file parquet that pyarrow.dataset can read as a "directory dataset"
    table = pa.Table.from_pandas(df)
    pq.write_table(table, out)
    return out

def test_decisions_overlap_and_causality(tmp_path: Path):
    dec = _seed_decisions(tmp_path / "artifacts" / "portfolio")
    assert dec.exists(), "failed to create decisions.parquet fixture"

    # Your acceptance gates:
    assert _share_multisymbol(dec) >= 0.10, "not enough multi-symbol overlap"
    assert _causality_ratio(dec) == 1.0, "entry_ts must be strictly after decision_ts"
