############################
# feature_engineering/calculators/adx.py
############################
"""Average Directional Index (ADX) calculator.

This implementation keeps the intermediate Series strictly *float* (using
``np.nan`` instead of Pandas' nullable scalar ``pd.NA``) so that Pandas'
rolling kernels receive numeric dtypes and avoid the
``pandas.errors.DataError: No numeric types to aggregate`` that can occur when
object dtypes sneak in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseCalculator, RollingCalculatorMixin


class ADXCalculator(RollingCalculatorMixin, BaseCalculator):
    """Compute the Average Directional Index (trend strength).

    Parameters
    ----------
    period : int, default ``14``
        Look‑back window used throughout the calculation.
    """

    def __init__(self, period: int = 14):
        self.period = period
        self.name = f"adx_{period}"
        # we need twice the period to get the first *fully* stable value
        self.lookback = period * 2

    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:  # noqa: D401
        """Return a DataFrame with a single *float32* ADX column."""

        # ------------------------------------------------------------------
        # Directional movement components
        # ------------------------------------------------------------------
        up_move   = df["high"].diff()
        down_move = -df["low"].diff()

        plus_dm  = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

        # ------------------------------------------------------------------
        # True range (smoothed over *period*)
        # ------------------------------------------------------------------
        tr = (
            pd.concat(
                [
                    df["high"] - df["low"],
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"]  - df["close"].shift()).abs(),
                ],
                axis=1,
            )
            .max(axis=1)
            .rolling(self.period)
            .sum()
        )
        tr = tr.replace(0, np.nan)  # keep dtype = float64

        # ------------------------------------------------------------------
        # Directional indicators
        # ------------------------------------------------------------------
        plus_di  = 100 * plus_dm.rolling(self.period).sum()  / tr
        minus_di = 100 * minus_dm.rolling(self.period).sum() / tr

        # ------------------------------------------------------------------
        # DX and ADX
        # ------------------------------------------------------------------
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
        dx = pd.to_numeric(dx, errors="coerce")  # ensure numeric dtype

        adx = dx.rolling(self.period, min_periods=self.period).mean()

        return pd.DataFrame({self.name: adx.astype("float32")})
