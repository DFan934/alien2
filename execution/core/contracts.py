# -----------------------------------------------------------------------------
# core/contracts.py
# -----------------------------------------------------------------------------
"""Pydantic data‑contracts shared across the trading system.

Schema evolution is tracked via `schema_version` so forward‑compat changes can
remain additive.  Side strings are normalised to upper‑case "BUY"/"SELL".
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, NonNegativeFloat, PositiveFloat, PositiveInt, root_validator, validator

SCHEMA_VERSION = "0.2.0"


class _Base(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION)


# --------------- Trade → Execution ------------------------------------------
class TradeSignal(_Base):
    id: str = Field(default_factory=lambda: uuid4().hex)
    symbol: str
    side: Literal["BUY", "SELL"]
    price: PositiveFloat
    confidence: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime

    # Feature payload (pre‑computed by PE / feature stage)
    atr: NonNegativeFloat
    vwap_dist: float  # % distance from VWAP (signed)
    ema_fast_dist: float  # % distance from 9‑EMA (signed)
    orderflow_delta: float
    regime: Literal["trend", "range", "volatile", "unknown"] = "unknown"

    # ---- normalisers -------------------------------------------------------
    @root_validator(pre=True)
    def _normalise_side(cls, values):  # noqa: D401
        if "side" in values and isinstance(values["side"], str):
            values["side"] = values["side"].upper()
        return values


# --------------- Execution → Broker -----------------------------------------
class OrderEvent(_Base):
    signal_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    qty: PositiveInt
    order_type: Literal["MKT", "LMT"] = "MKT"
    limit_px: Optional[float] = None
    stop_px: Optional[float] = None
    tp_tiers: List[float] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# --------------- Broker → Execution -----------------------------------------
class FillEvent(_Base):
    order_id: str
    signal_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    fill_px: float
    qty: PositiveInt
    timestamp: datetime


# --------------- SafetyFSM → Execution --------------------------------------
class SafetyAction(_Base):
    action: Literal["HALT", "RESUME", "SIZE_DOWN"]
    reason: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# --------------- Execution → Metrics ----------------------------------------
class BlotterRecord(_Base):
    signal_id: str
    fill_px: float
    stop_px: Optional[float]
    exit_px: Optional[float]
    pnl: float
    latency_ms: float
    safety_hit: Optional[str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
