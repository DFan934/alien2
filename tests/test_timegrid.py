import pandas as pd
import numpy as np

from feature_engineering.utils.timegrid import standardize_bars_to_grid


def _make(sym: str, start: str, n: int, step_s: int):
    ts = pd.date_range(start, periods=n, freq=f"{step_s}s", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": sym,
            "open": np.arange(n, dtype=float),
            "high": np.arange(n, dtype=float) + 1,
            "low": np.arange(n, dtype=float) - 1,
            "close": np.arange(n, dtype=float) + 0.5,
            "volume": np.ones(n, dtype=int),
        }
    )


def test_standardize_to_60s_grid():
    bby = _make("BBY", "1998-01-02 14:30:00", n=20, step_s=60)
    rrc = _make("RRC", "1998-01-02 14:30:00", n=6, step_s=240)

    bars = pd.concat([bby, rrc], ignore_index=True)
    out, audits = standardize_bars_to_grid(bars, freq="60s", expected_freq_s=60)

    # tz-aware UTC
    assert str(out["timestamp"].dtype).endswith(", UTC]")

    # no duplicates per (symbol,timestamp)
    assert not out.duplicated(subset=["symbol", "timestamp"]).any()

    # median delta close to 60s for both
    for sym in ["BBY", "RRC"]:
        ts = out[out["symbol"] == sym]["timestamp"].sort_values()
        med = ts.diff().dropna().dt.total_seconds().median()
        assert 55 <= float(med) <= 65

    # overlap should exist if their ranges overlap
    bby_set = set(out[out["symbol"] == "BBY"]["timestamp"])
    rrc_set = set(out[out["symbol"] == "RRC"]["timestamp"])
    assert len(bby_set.intersection(rrc_set)) > 0
