"""
Label generator used by PathClusterEngine.

Given a minute‑bar dataframe with columns
    ['timestamp','open','close', …],
produce a 1‑bar‑ahead **log‑return** label that aligns EXACTLY
with what EVEngine predicts at run‑time (next‑open / current‑open).

Usage
-----
#>>> from feature_engineering.labels.labeler import one_bar_ahead
#>>> y = one_bar_ahead(df_minute, horizon=1)   # Series aligned to df_minute.index
"""
from __future__ import annotations
import numpy as np
import pandas as pd

def one_bar_ahead(df: pd.DataFrame, *, horizon: int = 1) -> pd.Series:
    """
    Parameters
    ----------
    df : minute‑bar dataframe **sorted by timestamp**
    horizon : int, default 1
        Number of bars ahead; 1 = EVEngine’s “next open”.

    Returns
    -------
    y : pd.Series (same length, NaN for final `horizon` rows)
        log( open(t+H) / open(t) )
    """
    if "open" not in df.columns:
        raise KeyError("'open' column missing")

    open_px = df["open"].astype(float).values
    fwd = np.roll(open_px, -horizon)             # shift left
    y = np.log(fwd / open_px)
    y[-horizon:] = np.nan                       # final rows undefined
    return pd.Series(y, index=df.index, name="ret_fwd")

def one_bar_ahead_binary(df: pd.DataFrame, *, horizon: int = 1, threshold: float = 0.0) -> pd.Series:
    """
    Binary label: 1 iff log(open(t+H)/open(t)) > threshold, else 0.
    Returns a pd.Series of {0,1} with NaN for final `horizon` rows.
    """
    r = one_bar_ahead(df, horizon=horizon)  # log open→open
    y = (r > float(threshold)).astype("float").where(~r.isna(), np.nan)
    return y.rename("y")
