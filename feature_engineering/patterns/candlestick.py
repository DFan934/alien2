############################
# feature_engineering/patterns/candlestick.py
############################
from __future__ import annotations

import pandas as pd


def bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    body_prev = df["close"].shift() - df["open"].shift()
    body_curr = df["close"] - df["open"]
    cond = (body_prev < 0) & (body_curr > 0) & (df["open"] < df["close"].shift()) & (df["close"] > df["open"].shift())
    return cond.astype("int8")