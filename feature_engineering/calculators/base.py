############################
# feature_engineering/calculators/base.py
############################
"""Common interface + utility mixin for feature calculators."""
from __future__ import annotations

import pandas as pd
from typing import Protocol, runtime_checkable

@runtime_checkable
class Calculator(Protocol):
    """Minimal contract every calculator must fulfil."""

    name: str        # human‑readable unique key
    lookback: int    # periods required before first valid output (0 ok)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:  # noqa: D401
        """Return only new column(s). df is assumed time‑indexed + sorted."""


class RollingCalculatorMixin:
    """Helper to enforce sufficient lookback padding and NaN handling."""

    def _pad(self, series: pd.Series, pad_value=pd.NA) -> pd.Series:
        pad = pd.Series([pad_value] * self.lookback, index=series.index[: self.lookback])
        return pd.concat([pad, series.iloc[self.lookback :]])