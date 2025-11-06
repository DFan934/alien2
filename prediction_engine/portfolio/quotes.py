# prediction_engine/portfolio/quotes.py
from __future__ import annotations
from typing import Dict, Any
import pandas as pd
from .order_sim import QuoteStats

def estimate_quote_stats_from_rolling(
    row: "pd.Series",
    *,
    spread_bps: float = 2.0,  # half-spread ≈ 2 bps of price (tunable)
    adv_window: int = 20,
) -> QuoteStats:
    px = float(row.get("open", row.get("close", 0.0)))
    half_spread_usd = (spread_bps / 1e4) * px
    adv_shares = float(row.get("adv_shares", 0.0))
    if not adv_shares and "volume" in row and isinstance(row["volume"], (int, float)):
        # fallback: simple rolling average was precomputed upstream
        adv_shares = float(row["volume"])
    return QuoteStats(half_spread_usd=half_spread_usd, adv_shares=max(adv_shares, 1.0))
