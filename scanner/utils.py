# -----------------------------------------------------------------------------
# 1) scanner/utils.py  – time‑alignment helpers
# -----------------------------------------------------------------------------

"""Utility helpers for scanner package."""

from datetime import datetime, timezone, timedelta

__all__ = [
    "time_align_minute",
]

def time_align_minute(ts: datetime) -> datetime:
    """Snap *ts* to the start of the containing exchange‑minute (UTC).

    Works for both naïve and timezone‑aware datetimes.  Seconds and microseconds
    are zeroed so that replayed bars map deterministically onto the same bucket
    boundaries seen in live WebSocket feeds.
    """
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.replace(second=0, microsecond=0)
