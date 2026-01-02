import pandas as pd
from pathlib import Path
from pandas.api.types import is_datetime64tz_dtype

from scripts.run_backtest import _consolidate_phase4_outputs

def test_consolidate_parquet_timestamps_are_datetime(tmp_path: Path):
    # Arrange: build a faux artifacts_root with CSV inputs containing string timestamps
    root = tmp_path / "artifacts"
    fold = root / "fold_000"
    fold.mkdir(parents=True, exist_ok=True)

    # decisions.csv with string timestamps
    dec = pd.DataFrame({
        "timestamp": ["1999-01-05 09:30:00", "1999-01-05 09:31:00"],
        "symbol": ["RRC", "RRC"],
        "p_cal": [0.1, 0.2],
    })
    dec.to_csv(fold / "decisions.csv", index=False)

    # trades.csv with string entry/exit timestamps
    trd = pd.DataFrame({
        "entry_ts": ["1999-01-05 09:30:00", "1999-01-05 09:31:00"],
        "exit_ts":  ["1999-01-05 09:50:00", "1999-01-05 09:52:00"],
        "symbol": ["RRC", "RRC"],
        "qty": [10, 10],
        "realized_pnl_after_costs": [0.0, 0.01],
    })
    trd.to_csv(fold / "trades.csv", index=False)

    # Act
    dec_path, trd_path = _consolidate_phase4_outputs(root)

    # Assert: files exist
    assert dec_path is not None and dec_path.exists()
    assert trd_path is not None and trd_path.exists()

    # Read back and assert dtypes are datetime64[ns]
    dec_out = pd.read_parquet(dec_path)
    trd_out = pd.read_parquet(trd_path)

    assert "timestamp" in dec_out.columns
    assert is_datetime64_ns_dtype(dec_out["timestamp"]), f"decisions.timestamp dtype was {dec_out['timestamp'].dtype}"

    assert "entry_ts" in trd_out.columns and "exit_ts" in trd_out.columns
    assert is_datetime64_ns_dtype(trd_out["entry_ts"]), f"trades.entry_ts dtype was {trd_out['entry_ts'].dtype}"
    assert is_datetime64_ns_dtype(trd_out["exit_ts"]),  f"trades.exit_ts dtype was {trd_out['exit_ts'].dtype}"




import pandas as pd
from pathlib import Path
from pandas.api.types import is_datetime64_ns_dtype
from scripts.run_backtest import _consolidate_phase4_outputs

def test_consolidate_parquet_timestamps_mixed_tz(tmp_path: Path):
    root = tmp_path / "artifacts"
    fold = root / "fold_000"
    fold.mkdir(parents=True, exist_ok=True)

    # decisions with mixed tz (aware + naive) timestamps
    dec = pd.DataFrame({
        "timestamp": ["1999-01-05 09:30:00Z", "1999-01-05 09:31:00"],  # 'Z' = UTC-aware; second is naive
        "symbol": ["RRC", "RRC"],
        "p_cal": [0.1, 0.2],
    })
    dec.to_csv(fold / "decisions.csv", index=False)

    # trades with mixed tz
    trd = pd.DataFrame({
        "entry_ts": ["1999-01-05T09:30:00+00:00", "1999-01-05 09:31:00"],  # aware + naive
        "exit_ts":  ["1999-01-05 09:50:00Z",      "1999-01-05 09:52:00"],
        "symbol": ["RRC", "RRC"],
        "qty": [10, 10],
        "realized_pnl_after_costs": [0.0, 0.01],
    })
    trd.to_csv(fold / "trades.csv", index=False)

    dec_path, trd_path = _consolidate_phase4_outputs(root)
    assert dec_path is not None and trd_path is not None

    dec_out = pd.read_parquet(dec_path)
    trd_out = pd.read_parquet(trd_path)

    assert "timestamp" in dec_out.columns
    '''assert is_datetime64_ns_dtype(dec_out["timestamp"])

    assert is_datetime64_ns_dtype(trd_out["entry_ts"])
    assert is_datetime64_ns_dtype(trd_out["exit_ts"])
    '''
    assert is_datetime64tz_dtype(dec_out["timestamp"]), f"decisions.timestamp dtype was {dec_out['timestamp'].dtype}"
    assert str(dec_out["timestamp"].dtype) == "datetime64[ns, UTC]"

    assert is_datetime64tz_dtype(trd_out["entry_ts"]), f"trades.entry_ts dtype was {trd_out['entry_ts'].dtype}"
    assert str(trd_out["entry_ts"].dtype) == "datetime64[ns, UTC]"

    assert is_datetime64tz_dtype(trd_out["exit_ts"]), f"trades.exit_ts dtype was {trd_out['exit_ts'].dtype}"
    assert str(trd_out["exit_ts"].dtype) == "datetime64[ns, UTC]"





# prediction_engine/tests/test_consolidate_parquet_timestamps.py
import pandas as pd
import numpy as np
from pathlib import Path
from scripts.run_backtest import _consolidate_phase4_outputs

def test_consolidate_parquet_timestamps_are_datetime(tmp_path: Path):
    art = tmp_path / "artifacts" / "a2"
    fdir = art / "folds" / "fold_000"
    fdir.mkdir(parents=True, exist_ok=True)

    # Mixed tz on purpose: naive strings + tz-aware iso
    dec = pd.DataFrame({
        "timestamp": ["1999-01-05 09:30:00", "1999-01-05T09:31:00Z"],
        "decision_ts": ["1999-01-05 09:30:00", "1999-01-05T09:31:00Z"],
        "symbol": ["RRC", "BBY"],
        "p_cal": [0.1, 0.2],
    })
    trd = pd.DataFrame({
        "entry_ts": ["1999-01-05 09:31:00", "1999-01-05T09:32:00Z"],
        "exit_ts":  ["1999-01-05 09:32:00", "1999-01-05T09:33:00Z"],
        "entry_price": [10.0, 10.1],
        "exit_price":  [10.2, 10.0],
        "qty": [100, 50],
        "realized_pnl": [20.0, -5.0],
    })

    # Write as fold outputs (csv to exercise both readers)
    dec.to_csv(fdir / "decisions.csv", index=False)
    trd.to_parquet(fdir / "trades.parquet", index=False)

    dec_path, trd_path = _consolidate_phase4_outputs(art)

    # Should succeed without tz-mix errors and produce parquet files
    assert dec_path and trd_path
    dec2 = pd.read_parquet(dec_path)
    trd2 = pd.read_parquet(trd_path)

    # Check tz-aware UTC
    for c in ("timestamp", "decision_ts"):
        if c in dec2.columns:
            assert str(dec2[c].dtype).startswith("datetime64[ns, UTC]")

    for c in ("entry_ts", "exit_ts"):
        if c in trd2.columns:
            assert str(trd2[c].dtype).startswith("datetime64[ns, UTC]")
