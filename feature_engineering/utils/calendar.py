# feature_engineering/utils/calendar.py
from __future__ import annotations
import pandas as pd
import numpy as np

_NYSE_OPEN_UTC = (14, 30)   # 9:30 ET
_NYSE_CLOSE_UTC = (21, 00)  # 16:00 ET

def _as_utc_ts(ts: pd.Series) -> pd.Series:
    if ts.dt.tz is None:
        # assume naive is already UTC; enforce tz-naive UTC in pipeline before calling
        return ts
    return ts.dt.tz_convert("UTC")

def session_id(ts: pd.Series) -> pd.Series:
    """Return a date-like 'session key' for each timestamp (UTC date)."""
    ts_utc = _as_utc_ts(ts)
    return ts_utc.dt.floor("D")

def is_rth(ts: pd.Series) -> pd.Series:
    """True if timestamp falls inside NYSE regular-hours session (simple bound)."""
    ts_utc = _as_utc_ts(ts)
    hhmm = ts_utc.dt.hour * 60 + ts_utc.dt.minute
    open_m = _NYSE_OPEN_UTC[0] * 60 + _NYSE_OPEN_UTC[1]
    close_m = _NYSE_CLOSE_UTC[0] * 60 + _NYSE_CLOSE_UTC[1]
    return (hhmm >= open_m) & (hhmm < close_m)

def minutes_since_open(ts: pd.Series) -> pd.Series:
    """Minute index within the RTH session (0-based). Values <0 or >= len(session) indicate pre/post."""
    ts_utc = _as_utc_ts(ts)
    open_m = _NYSE_OPEN_UTC[0] * 60 + _NYSE_OPEN_UTC[1]
    return (ts_utc.dt.hour * 60 + ts_utc.dt.minute) - open_m

def is_early_close(session_dates: pd.Series) -> pd.Series:
    """Optional hook: mark known early-close days (skeleton; wire real calendar if desired)."""
    # TODO: integrate trading_calendars for authoritative early-close holidays.
    return pd.Series(False, index=session_dates.index)


def slots_since_open(ts: pd.Series, bar_seconds: int = 60) -> pd.Series:
    """
    Return the zero-based index of the *slot* within the RTH session,
    where slot width == bar_seconds. For 1-min bars, this is minutes-since-open.
    For 5-min bars, this increments every 5 minutes.
    """
    ts_utc = _as_utc_ts(ts)
    open_m = _NYSE_OPEN_UTC[0] * 60 + _NYSE_OPEN_UTC[1]
    minutes_from_open = (ts_utc.dt.hour * 60 + ts_utc.dt.minute) - open_m
    # Convert minutes to slots by cadence
    slot = np.floor_divide(minutes_from_open * 60, int(bar_seconds))
    return slot.astype("int32")
