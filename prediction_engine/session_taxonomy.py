# prediction_engine/session_taxonomy.py  — batch day‑label generator
# ---------------------------------------------------------------------
"""Batch classifier that labels every trading day as TREND / RANGE / VOLATILE.

* Re‑uses the `label_days()` routine from `market_regime.py` so that the
  historical backfill and the streaming detector are consistent.
* CLI writes `day_labels.csv` next to the supplied OHLCV file.
"""

from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

# Support both package execution (`python -m prediction_engine.session_taxonomy`)
# and direct script run (`python prediction_engine/session_taxonomy.py`).
try:
    from .market_regime import label_days, RegimeParams  # type: ignore
except ImportError:  # fallback when run as a stand‑alone script
    from prediction_engine.market_regime import label_days, RegimeParams

__all__ = ["generate_labels"]

# ------------------------------------------------------------------

def generate_labels(ohlcv_path: Path, *, out_csv: Path | None = None, params: RegimeParams | None = None) -> pd.Series:
    """Return a Series of regime labels (index = date).  Optionally writes CSV."""
    df = _load_frame(ohlcv_path)
    labels = label_days(df, params)
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        labels.to_csv(out_csv, header=["regime"], date_format="%Y-%m-%d")
        print(f"[session_taxonomy] wrote {out_csv}")
    return labels

# ------------------------------------------------------------------
# helpers -----------------------------------------------------------

def _load_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, parse_dates=[0])
    df = df.rename(columns=str.lower)
    req = {"high", "low", "close"}
    if not req.issubset(df.columns):
        raise ValueError(f"{path} must contain columns {req}")
    df = df.set_index(df.columns[0])  # date index
    return df

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Defaults & self‑test ------------------------------------------------

DEFAULT_OHLCV_PATH = Path("store/daily_ohlcv.parquet")


def _self_test() -> None:  # pragma: no cover – quick demo
    import yfinance as yf
    df = yf.download("SPY", period="1y", interval="1d").rename(columns=str.lower)
    tmp = Path("/tmp/spy_1d.parquet")
    df.to_parquet(tmp)
    generate_labels(tmp, out_csv=tmp.with_name("day_labels.csv"))


# ------------------------------------------------------------------
# Entry‑point – runnable via green ▶ button -------------------------

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        ohlcv = Path(sys.argv[1])
        out = ohlcv.with_name("day_labels.csv")
        generate_labels(ohlcv, out_csv=out)
    elif DEFAULT_OHLCV_PATH.exists():
        generate_labels(DEFAULT_OHLCV_PATH, out_csv=DEFAULT_OHLCV_PATH.with_name("day_labels.csv"))
    else:
        print("[session_taxonomy] No OHLCV file specified; running self‑test.")
        _self_test()
