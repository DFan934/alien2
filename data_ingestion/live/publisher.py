# data_ingestion/live/publisher.py
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from data_ingestion.utils import logger
from data_ingestion.live.bar_finalizer import BarFinalizer
from data_ingestion.live.alpaca_ws_client import AlpacaWSClient

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
except Exception as e:  # pragma: no cover
    pa = None  # type: ignore
    pq = None  # type: ignore


def _parse_rfc3339_any(s: Any) -> datetime:
    """
    Alpaca bar event timestamp field often comes as RFC3339 string (with Z).
    Return tz-aware UTC datetime.
    """
    if isinstance(s, datetime):
        if s.tzinfo is None:
            raise ValueError("naive datetime not allowed")
        return s.astimezone(timezone.utc)
    if not isinstance(s, str) or not s:
        raise ValueError(f"bad timestamp: {s!r}")
    s2 = s.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s2)
    if dt.tzinfo is None:
        raise ValueError("timestamp must be tz-aware")
    return dt.astimezone(timezone.utc)


def _alpaca_bar_to_ohlcv(ev: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Accepts Alpaca market data bar event formats.

    Common v2 fields:
      T: 'b' (bar)
      S: symbol
      t: timestamp (RFC3339)
      o,h,l,c,v: OHLCV

    Returns dict with: symbol, ts(datetime UTC), open, high, low, close, volume
    """
    try:
        typ = ev.get("T") or ev.get("type")
        if typ not in ("b", "bar", "BAR"):
            return None

        sym = ev.get("S") or ev.get("symbol")
        ts = ev.get("t") or ev.get("timestamp")

        o = ev.get("o") if "o" in ev else ev.get("open")
        h = ev.get("h") if "h" in ev else ev.get("high")
        l = ev.get("l") if "l" in ev else ev.get("low")
        c = ev.get("c") if "c" in ev else ev.get("close")
        v = ev.get("v") if "v" in ev else ev.get("volume")

        if sym is None or ts is None:
            return None

        return {
            "symbol": str(sym).upper(),
            "ts": _parse_rfc3339_any(ts),
            "open": float(o),
            "high": float(h),
            "low": float(l),
            "close": float(c),
            "volume": float(v),
        }
    except Exception:
        return None


class _ParquetAppender:
    """
    Append-only parquet writer using pyarrow. Keeps a single file open.

    This gives you the exact Step 5 artifact names:
      bars_raw.parquet
      bars_final.parquet
    """
    def __init__(self, path: Path) -> None:
        if pq is None or pa is None:  # pragma: no cover
            raise ImportError("pyarrow is required for parquet writing.")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer: Optional[pq.ParquetWriter] = None

    def append_df(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, table.schema, compression="snappy")
        self._writer.write_table(table)

    def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass
        self._writer = None


@dataclass
class PublisherConfig:
    # soft flush cadence (write small batches frequently)
    flush_every_n_raw: int = 200
    flush_every_n_final: int = 200
    emit_gaps: bool = True


class MarketDataPublisher:
    """
    Step 5: MarketData client -> BarFinalizer -> ScannerLoop queue injection.

    - Consumes raw Alpaca WS events
    - Writes bars_raw.parquet (raw flattened rows)
    - Finalizes minute bars (and gaps) via BarFinalizer
    - Writes bars_final.parquet (canonical finalized rows)
    - Pushes finalized batches to bar_queue (list[dict]) for scanner/live_loop.py
    """

    def __init__(
        self,
        *,
        cfg: Dict[str, Any],
        out_dir: Path,
        bar_queue: "asyncio.Queue",
        symbols: list[str],
        publisher_cfg: Optional[PublisherConfig] = None,
        ws_client: Optional[AlpacaWSClient] = None,
        bar_finalizer: Optional[BarFinalizer] = None,
    ) -> None:
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.bar_queue = bar_queue
        self.symbols = [str(s).upper() for s in symbols]

        self.pcfg = publisher_cfg or PublisherConfig(
            flush_every_n_raw=int(((cfg.get("live") or {}).get("flush_every_n_raw", 200))),
            flush_every_n_final=int(((cfg.get("live") or {}).get("flush_every_n_final", 200))),
            emit_gaps=bool(((cfg.get("live") or {}).get("emit_gaps", True))),
        )

        self.ws = ws_client or AlpacaWSClient(cfg)
        self.finalizer = bar_finalizer or BarFinalizer(freq="60s", emit_gaps=self.pcfg.emit_gaps)

        self._raw_writer = _ParquetAppender(self.out_dir / "bars_raw.parquet")
        self._final_writer = _ParquetAppender(self.out_dir / "bars_final.parquet")

        self._raw_buf: list[Dict[str, Any]] = []
        self._final_buf: list[Dict[str, Any]] = []

    async def run(self, *, max_raw_messages: Optional[int] = None) -> None:
        """
        Run until cancelled (production) or until max_raw_messages (tests).
        """
        raw_count = 0
        try:
            async for ev in self.ws.stream_raw(symbols=self.symbols):
                raw_count += 1

                # ---------- raw artifact ----------
                self._raw_buf.append(self._raw_flat_row(ev))
                if len(self._raw_buf) >= self.pcfg.flush_every_n_raw:
                    self._flush_raw()

                # ---------- finalize + queue ----------
                parsed = _alpaca_bar_to_ohlcv(ev)
                if parsed is not None:
                    finalized = self.finalizer.ingest_bar_update(
                        symbol=parsed["symbol"],
                        ts=parsed["ts"],
                        open=parsed["open"],
                        high=parsed["high"],
                        low=parsed["low"],
                        close=parsed["close"],
                        volume=parsed["volume"],
                    )
                    if finalized:
                        # artifact write
                        self._final_buf.extend(finalized)
                        if len(self._final_buf) >= self.pcfg.flush_every_n_final:
                            self._flush_final()

                        # queue injection (ScannerLoop expects a batch: list[dict])
                        await self.bar_queue.put(finalized)

                if max_raw_messages is not None and raw_count >= max_raw_messages:
                    break

        finally:
            # final flushes
            self._flush_raw()
            self._flush_final()
            self._raw_writer.close()
            self._final_writer.close()
            try:
                await self.ws.close()
            except Exception:
                pass

            logger.info("[Publisher] stopped. raw=%d", raw_count)

    def _raw_flat_row(self, ev: Dict[str, Any]) -> Dict[str, Any]:
        # keep it tolerant; we don't enforce a strict schema for raw
        sym = ev.get("S") or ev.get("symbol") or ""
        ts = ev.get("t") or ev.get("timestamp") or None
        received = ev.get("_received_ts_utc") or datetime.now(timezone.utc).isoformat()

        return {
            "received_ts_utc": received,
            "event_type": ev.get("T") or ev.get("type") or "",
            "symbol": str(sym).upper() if sym else "",
            "event_ts": str(ts) if ts is not None else "",
            "raw": str(ev)[:2000],  # cap to avoid gigantic rows
        }

    def _flush_raw(self) -> None:
        if not self._raw_buf:
            return
        df = pd.DataFrame(self._raw_buf)
        self._raw_writer.append_df(df)
        self._raw_buf = []

    def _flush_final(self) -> None:
        if not self._final_buf:
            return
        df = pd.DataFrame(self._final_buf)
        # ensure stable ordering for inspection
        if "timestamp" in df.columns and "symbol" in df.columns:
            df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        self._final_writer.append_df(df)
        self._final_buf = []
