# data_ingestion/contracts.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional
from uuid import uuid4

from pydantic import (
    BaseModel,
    Field,
    PositiveFloat,
    NonNegativeFloat,
    PositiveInt,
    model_validator,   # <-- add
)

SCHEMA_VERSION_LIVE_INGEST = "0.1.0"

EventType = Literal["BAR", "QUOTE", "TRADE"]
BarTimeframe = Literal["1Min"]


class LiveBar(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION_LIVE_INGEST)

    symbol: str
    timeframe: BarTimeframe = "1Min"

    ts_start_utc: datetime
    ts_end_utc: datetime

    open: PositiveFloat
    high: PositiveFloat
    low: PositiveFloat
    close: PositiveFloat
    volume: NonNegativeFloat

    vwap: Optional[PositiveFloat] = None
    trade_count: Optional[PositiveInt] = None

    source: str = "unknown"
    is_final: bool = True
    received_ts_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _validate_livebar(self) -> "LiveBar":
        # ---- tz-awareness (post-parse; works even if inputs were strings) ----
        for k in ("ts_start_utc", "ts_end_utc", "received_ts_utc"):
            dt = getattr(self, k)
            if dt.tzinfo is None:
                raise ValueError(f"{k} must be timezone-aware (UTC)")

        # ---- time ordering ----
        if not (self.ts_end_utc > self.ts_start_utc):
            raise ValueError("ts_end_utc must be strictly after ts_start_utc")

        # ---- OHLC invariants ----
        if self.low > self.high:
            raise ValueError("low must be <= high")

        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high must be >= max(open, close, low)")

        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low must be <= min(open, close, high)")

        return self


class MarketDataEvent(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION_LIVE_INGEST)

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    event_type: EventType
    symbol: str

    event_ts_utc: datetime
    received_ts_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    bar: Optional[LiveBar] = None
    raw: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def _validate_event(self) -> "MarketDataEvent":
        # tz-awareness (post-parse)
        for k in ("event_ts_utc", "received_ts_utc"):
            dt = getattr(self, k)
            if dt.tzinfo is None:
                raise ValueError(f"{k} must be timezone-aware (UTC)")

        # conditional payload requirement
        if self.event_type == "BAR":
            if self.bar is None:
                raise ValueError("bar payload must be present when event_type == 'BAR'")
            # optional but nice: enforce symbol consistency
            if self.bar.symbol != self.symbol:
                raise ValueError("MarketDataEvent.symbol must match bar.symbol")

        return self
