import pandas as pd
import pytest
from scanner.detectors import CompositeDetector

class _DetA:
    def __call__(self, df: pd.DataFrame) -> pd.Series:
        # True on rows 0 and 2
        return pd.Series([True, False, True], index=df.index)

class _DetB:
    def __call__(self, df: pd.DataFrame) -> pd.Series:
        # True on rows 1 and 2
        return pd.Series([False, True, True], index=df.index)

@pytest.mark.asyncio
async def test_and_vs_or():
    df = pd.DataFrame(index=pd.RangeIndex(3))
    cd_and = CompositeDetector([_DetA(), _DetB()], mode="AND")
    cd_or  = CompositeDetector([_DetA(), _DetB()], mode="OR")

    res_and = await cd_and(df)
    res_or  = await cd_or(df)

    # AND should only pass row 2
    assert res_and.tolist() == [False, False, True]
    # OR should pass all rows (0,1,2)
    assert res_or.tolist()  == [True, True, True]
    # AND hit-count < OR hit-count
    assert res_and.sum() < res_or.sum()

@pytest.mark.asyncio
async def test_default_is_and():
    df = pd.DataFrame(index=pd.RangeIndex(3))
    default_cd = CompositeDetector([_DetA(), _DetB()])  # no mode specified
    res = await default_cd(df)
    assert res.tolist() == [False, False, True]
