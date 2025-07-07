############################
# feature_engineering/calculators/base.py
############################
from __future__ import annotations
from typing import Protocol, runtime_checkable
import pandas as pd

class BaseCalculator:
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:   # noqa: D401
        new_cols = self.transform(df)       # only the calculator’s features
        return pd.concat([df, new_cols], axis=1)

@runtime_checkable
class Calculator(Protocol):
    name: str
    lookback: int
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...
    # NEW
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:  # noqa: D401
        return self.transform(df)

class RollingCalculatorMixin:
    def _pad(self, series: pd.Series, pad_value=pd.NA) -> pd.Series:
        pad = pd.Series([pad_value] * self.lookback, index=series.index[: self.lookback])
        return pd.concat([pad, series.iloc[self.lookback :]])
