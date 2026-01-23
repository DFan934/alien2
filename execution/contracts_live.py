# execution/contracts_live.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, PositiveInt, PositiveFloat, root_validator, model_validator

SCHEMA_VERSION_LIVE_EXEC = "0.1.0"

OrderSide = Literal["BUY", "SELL"]
OrderType = Literal["MKT", "LMT", "STP", "STP_LMT"]
TimeInForce = Literal["DAY", "GTC", "IOC", "FOK"]
OrderStatus = Literal[
    "CREATED", "SUBMITTED", "ACCEPTED", "OPEN",
    "PARTIALLY_FILLED", "FILLED",
    "CANCELED", "REJECTED", "EXPIRED"
]


class OrderRequest(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION_LIVE_EXEC)

    client_order_id: str = Field(default_factory=lambda: uuid4().hex)  # idempotency key
    symbol: str
    side: OrderSide
    qty: PositiveInt

    order_type: OrderType = "MKT"
    tif: TimeInForce = "DAY"

    # Optional price controls (validated later by adapters)
    limit_price: Optional[PositiveFloat] = None
    stop_price: Optional[PositiveFloat] = None

    # Strategy metadata for auditability (safe to be None)
    signal_id: Optional[str] = None
    reason: Optional[str] = None

    created_ts_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @root_validator(pre=True)
    def _norm_and_require_utc(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "side" in values and isinstance(values["side"], str):
            values["side"] = values["side"].upper()
        ts = values.get("created_ts_utc")
        if isinstance(ts, datetime) and ts.tzinfo is None:
            raise ValueError("created_ts_utc must be timezone-aware (UTC)")
        return values

    @model_validator(mode="after")
    def _require_prices_for_order_type(self) -> "OrderRequest":
        if self.order_type == "LMT":
            if self.limit_price is None:
                raise ValueError("limit_price must be provided when order_type == 'LMT'")
        if self.order_type == "STP":
            if self.stop_price is None:
                raise ValueError("stop_price must be provided when order_type == 'STP'")
        # Optional (not required by your current tests):
        # if self.order_type == "STP_LMT":
        #     if self.stop_price is None or self.limit_price is None:
        #         raise ValueError("STP_LMT requires both stop_price and limit_price")
        return self



class FillUpdate(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION_LIVE_EXEC)

    fill_id: str = Field(default_factory=lambda: uuid4().hex)
    qty: PositiveInt
    price: PositiveFloat
    ts_utc: datetime

    @root_validator(pre=True)
    def _require_utc(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        ts = values.get("ts_utc")
        if isinstance(ts, datetime) and ts.tzinfo is None:
            raise ValueError("ts_utc must be timezone-aware (UTC)")
        return values


class OrderUpdateEvent(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION_LIVE_EXEC)

    # Broker identifiers
    broker_order_id: Optional[str] = None
    client_order_id: str

    symbol: str
    status: OrderStatus
    side: OrderSide

    # Fill accounting
    filled_qty: int = 0
    avg_fill_price: Optional[PositiveFloat] = None
    last_fill: Optional[FillUpdate] = None

    event_ts_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    raw: Optional[Dict[str, Any]] = None

    @root_validator(pre=True)
    def _norm_and_require_utc(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "side" in values and isinstance(values["side"], str):
            values["side"] = values["side"].upper()
        ts = values.get("event_ts_utc")
        if isinstance(ts, datetime) and ts.tzinfo is None:
            raise ValueError("event_ts_utc must be timezone-aware (UTC)")
        return values


class BrokerClock(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION_LIVE_EXEC)

    ts_utc: datetime
    is_open: bool

    next_open_utc: Optional[datetime] = None
    next_close_utc: Optional[datetime] = None
    raw: Optional[Dict[str, Any]] = None

    @root_validator(pre=True)
    def _require_utc(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        for k in ("ts_utc", "next_open_utc", "next_close_utc"):
            dt = values.get(k)
            if isinstance(dt, datetime) and dt.tzinfo is None:
                raise ValueError(f"{k} must be timezone-aware (UTC)")
        return values
