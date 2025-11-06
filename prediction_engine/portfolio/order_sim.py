# prediction_engine/portfolio/order_sim.py
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Dict, Any

@dataclass(frozen=True)
class QuoteStats:
    """Minimal L1 stand-in: half-spread in USD/share and ADV in shares."""
    half_spread_usd: float = 0.015
    adv_shares: float = 1_000_000.0

@dataclass(frozen=True)
class EntryResult:
    filled_qty: float
    fill_ratio: float
    entry_price: float
    entry_ts: "pd.Timestamp"
    half_spread_usd: float
    adv_frac: float   # participation vs ADV (filled_qty / adv_shares)

@dataclass(frozen=True)
class ExitResult:
    exit_price: float
    exit_ts: "pd.Timestamp"
    bars_held: int
    realized_pnl: float

def _signed(x: float, side: int) -> float:
    return float(abs(x) * (1 if side >= 0 else -1))

def simulate_entry(
    *,
    decision_row: "pd.Series",
    next_open_price: float,
    next_open_ts: "pd.Timestamp",
    quote: QuoteStats,
    rules: Dict[str, Any],
) -> EntryResult:
    """
    Partial fills: cap at α * bar_volume (α=max_participation).
    Gap band for MOO: add +/- half-spread when adverse gap (simple band).
    """
    side = 1 if float(decision_row.get("target_qty", 0.0)) >= 0 else -1
    target_qty = float(decision_row.get("target_qty", 0.0))
    bar_vol = float(decision_row.get("bar_volume", rules.get("default_bar_volume", 1_000_000)))
    alpha = float(rules.get("max_participation", 0.1))  # 10% of bar volume by default
    cap = max(0.0, alpha * bar_vol)

    filled_qty_abs = min(abs(target_qty), cap)
    fill_ratio = 0.0 if abs(target_qty) < 1e-9 else (filled_qty_abs / abs(target_qty))
    filled_qty = _signed(filled_qty_abs, side)

    # Gap band: for MOO, push price by half-spread against the trader
    use_moo_band = bool(rules.get("moo_gap_band", True))
    entry_px = float(next_open_price)
    if use_moo_band and abs(filled_qty) > 0:
        entry_px = entry_px + (quote.half_spread_usd * (1 if side > 0 else -1))

    adv_frac = 0.0 if quote.adv_shares <= 0 else (filled_qty_abs / quote.adv_shares)
    return EntryResult(
        filled_qty=filled_qty,
        fill_ratio=fill_ratio,
        entry_price=entry_px,
        entry_ts=next_open_ts,
        half_spread_usd=quote.half_spread_usd,
        adv_frac=adv_frac,
    )

def simulate_exit(
    *,
    position_row: "pd.Series",
    exit_open_price: float,
    exit_open_ts: "pd.Timestamp",
    bars_held: int,
    quote: QuoteStats,
    rules: Dict[str, Any],
) -> ExitResult:
    """
    Exit after H bars at open; apply same gap band convention as entries.
    """
    qty = float(position_row["filled_qty"])
    side = 1 if qty >= 0 else -1

    use_moc_band = bool(rules.get("moc_gap_band", True))
    exit_px = float(exit_open_price)
    if use_moc_band and abs(qty) > 0:
        exit_px = exit_px - (quote.half_spread_usd * (1 if side > 0 else -1))

    entry_px = float(position_row["entry_price"])
    realized_pnl = (exit_px - entry_px) * qty  # signed qty carries the side
    return ExitResult(
        exit_price=exit_px,
        exit_ts=exit_open_ts,
        bars_held=int(bars_held),
        realized_pnl=realized_pnl,
    )
