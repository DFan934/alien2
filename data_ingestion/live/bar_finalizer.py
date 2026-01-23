from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _to_utc_ts(ts: Any) -> pd.Timestamp:
    """
    Normalize any datetime-like input to a tz-aware UTC pandas Timestamp.
    - If ts is naive datetime -> raise ValueError (we enforce tz-aware upstream)
    - If ts has tz -> convert to UTC
    """
    if isinstance(ts, pd.Timestamp):
        if ts.tz is None:
            raise ValueError("timestamp must be tz-aware (got naive pd.Timestamp)")
        return ts.tz_convert("UTC")

    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            raise ValueError("timestamp must be tz-aware (got naive datetime)")
        return pd.Timestamp(ts.astimezone(timezone.utc))

    # Let pandas try; but enforce tz-awareness after conversion.
    t = pd.to_datetime(ts)
    if isinstance(t, pd.DatetimeIndex):
        raise ValueError("expected scalar timestamp, got DatetimeIndex")
    if t.tz is None:
        raise ValueError("timestamp must be tz-aware (got naive value after pd.to_datetime)")
    return t.tz_convert("UTC")


def _floor_to_minute(ts_utc: pd.Timestamp, freq: str) -> pd.Timestamp:
    # freq expected "60s" for now; keep generic.
    return ts_utc.floor(freq)


def _canonical_bar_row(
    symbol: str,
    minute_ts_utc: pd.Timestamp,
    o: Optional[float],
    h: Optional[float],
    l: Optional[float],
    c: Optional[float],
    v: Optional[float],
    is_gap: bool,
) -> Dict[str, Any]:
    # Keep schema aligned with historical OHLCV + symbol, plus gap flags for live.
    row = {
        "timestamp": minute_ts_utc,            # tz-aware UTC
        "symbol": str(symbol).upper(),
        "open": np.nan if o is None else float(o),
        "high": np.nan if h is None else float(h),
        "low": np.nan if l is None else float(l),
        "close": np.nan if c is None else float(c),
        "volume": 0 if (v is None or is_gap) else int(v),
        "bar_present": 0 if is_gap else 1,
        "is_gap": bool(is_gap),
    }
    return row


@dataclass
class _SymbolState:
    # Buffer of minute -> last-seen OHLCV for that minute
    buf: Dict[pd.Timestamp, Dict[str, Any]]
    latest_seen_minute: Optional[pd.Timestamp]
    last_emitted_minute: Optional[pd.Timestamp]


class BarFinalizer:
    """
    Live bar finalizer that:
    - normalizes timestamps to UTC
    - floors timestamps onto a fixed minute grid
    - emits finalized rows up to a watermark (latest_seen_minute - 1 minute)
    - emits GAP rows for missing minutes between last emission and watermark

    IMPORTANT BEHAVIOR:
    - We do not emit the current "latest" minute until we see a strictly later minute,
      so you don't leak incomplete bars into downstream logic.
    """

    def __init__(self, freq: str = "60s", emit_gaps: bool = True) -> None:
        self.freq = str(freq)
        self.emit_gaps = bool(emit_gaps)
        self._states: Dict[str, _SymbolState] = {}

    def _state(self, symbol: str) -> _SymbolState:
        sym = str(symbol).upper()
        if sym not in self._states:
            self._states[sym] = _SymbolState(buf={}, latest_seen_minute=None, last_emitted_minute=None)
        return self._states[sym]

    def ingest_bar_update(
        self,
        *,
        symbol: str,
        ts: Any,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> List[Dict[str, Any]]:
        """
        Ingest a bar update (could arrive out-of-order).
        Returns a list of finalized rows (including possible GAP rows) in ascending timestamp order.
        """
        sym = str(symbol).upper()
        st = self._state(sym)

        ts_utc = _to_utc_ts(ts)
        minute_ts = _floor_to_minute(ts_utc, self.freq)

        # Store / overwrite last-seen values for that minute
        st.buf[minute_ts] = {
            "open": float(open),
            "high": float(high),
            "low": float(low),
            "close": float(close),
            "volume": float(volume),
        }

        # Advance watermark
        if st.latest_seen_minute is None or minute_ts > st.latest_seen_minute:
            st.latest_seen_minute = minute_ts

        return self._emit_ready(sym)

    def _emit_ready(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Emit rows up to watermark = latest_seen_minute - freq (i.e., everything strictly older than latest).
        If emit_gaps=True, fill missing minutes with GAP rows.
        """
        st = self._state(symbol)
        if st.latest_seen_minute is None:
            return []

        # Watermark: we only finalize minutes strictly before the latest observed minute
        watermark = st.latest_seen_minute - pd.Timedelta(self.freq)

        # Nothing to emit if watermark is before/equal last emitted
        if st.last_emitted_minute is not None and watermark <= st.last_emitted_minute:
            return []

        # Determine first minute to consider emitting
        if st.last_emitted_minute is None:
            if not st.buf:
                return []
            start_minute = min(st.buf.keys())
            # But if start_minute itself is newer than watermark, still nothing to emit.
            if start_minute > watermark:
                return []
        else:
            start_minute = st.last_emitted_minute + pd.Timedelta(self.freq)

        minutes = list(pd.date_range(start=start_minute, end=watermark, freq=self.freq, tz="UTC"))
        out: List[Dict[str, Any]] = []

        for m in minutes:
            if m in st.buf:
                b = st.buf.pop(m)
                out.append(_canonical_bar_row(
                    symbol=symbol,
                    minute_ts_utc=m,
                    o=b.get("open"),
                    h=b.get("high"),
                    l=b.get("low"),
                    c=b.get("close"),
                    v=b.get("volume"),
                    is_gap=False,
                ))
            else:
                if self.emit_gaps:
                    out.append(_canonical_bar_row(
                        symbol=symbol,
                        minute_ts_utc=m,
                        o=None, h=None, l=None, c=None, v=None,
                        is_gap=True,
                    ))

            st.last_emitted_minute = m

        return out

    def flush(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Flush everything currently buffered for a symbol, emitting in timestamp order.
        NOTE: This will emit the most recent minute too (used at shutdown / end-of-session).
        """
        sym = str(symbol).upper()
        st = self._state(sym)
        if not st.buf:
            return []

        # For flush, set watermark to max buffered minute so everything is emitted
        max_m = max(st.buf.keys())
        st.latest_seen_minute = max_m + pd.Timedelta(self.freq)  # hack: so watermark becomes max_m
        return self._emit_ready(sym)
