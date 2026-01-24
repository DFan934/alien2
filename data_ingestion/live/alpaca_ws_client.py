# data_ingestion/live/alpaca_ws_client.py
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Iterable, Optional

from data_ingestion.utils import logger

try:
    import websockets  # type: ignore
except Exception as e:  # pragma: no cover
    websockets = None  # type: ignore


def _env_expand(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if s.startswith("${") and s.endswith("}"):
        key = s[2:-1].strip()
        return os.environ.get(key)
    return s


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class AlpacaWSConfig:
    ws_url: str
    key_id: str
    secret_key: str
    # Bars subscription field name in Alpaca v2 stream: "bars"
    # We only implement bars in Step 5.
    heartbeat_timeout_s: float = 30.0


class AlpacaWSClient:
    """
    Minimal Alpaca Market Data websocket client for Step 5.

    Produces *raw* updates (decoded JSON) as Python dicts.

    Notes:
    - Uses Alpaca market-data websocket (NOT trading websocket).
    - Default URL should come from config.yaml; we provide a sane fallback.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        live_cfg = (cfg.get("live") or {})
        broker_cfg = (cfg.get("broker") or {})
        alpaca_cfg = (broker_cfg.get("alpaca") or {})

        ws_url = str(live_cfg.get("market_data_ws_url") or "wss://stream.data.alpaca.markets/v2/iex").strip()
        key_id = _env_expand(alpaca_cfg.get("key_id")) or ""
        secret_key = _env_expand(alpaca_cfg.get("secret_key")) or ""

        if not key_id or not secret_key:
            raise ValueError(
                "Missing Alpaca credentials for websocket auth. "
                "Set broker.alpaca.key_id/secret_key (supports ${ENV_VAR})."
            )

        self._cfg = AlpacaWSConfig(
            ws_url=ws_url,
            key_id=key_id,
            secret_key=secret_key,
            heartbeat_timeout_s=float(live_cfg.get("ws_heartbeat_timeout_s", 30.0)),
        )

        self._ws = None
        self._connected = False

    async def connect(self) -> None:
        if websockets is None:  # pragma: no cover
            raise ImportError(
                "Missing dependency 'websockets'. Install it (pip install websockets) to use AlpacaWSClient."
            )
        self._ws = await websockets.connect(self._cfg.ws_url, ping_interval=None)
        self._connected = True

        # Auth
        await self._send({"action": "auth", "key": self._cfg.key_id, "secret": self._cfg.secret_key})
        # Consume auth response(s) (Alpaca usually replies with a list of status messages)
        auth_msgs = await self._recv_json()
        logger.info("[WS] auth response: %s", auth_msgs)

    async def close(self) -> None:
        self._connected = False
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None

    async def _send(self, obj: Dict[str, Any]) -> None:
        if not self._ws:
            raise RuntimeError("websocket not connected")
        await self._ws.send(json.dumps(obj))

    async def _recv_json(self) -> Any:
        if not self._ws:
            raise RuntimeError("websocket not connected")
        raw = await asyncio.wait_for(self._ws.recv(), timeout=self._cfg.heartbeat_timeout_s)
        return json.loads(raw)

    async def subscribe_bars(self, symbols: Iterable[str]) -> None:
        syms = [str(s).upper() for s in symbols]
        await self._send({"action": "subscribe", "bars": syms})
        sub_msgs = await self._recv_json()
        logger.info("[WS] subscribe response: %s", sub_msgs)

    async def stream_raw(self, *, symbols: list[str]) -> AsyncIterator[Dict[str, Any]]:
        """
        Yields raw event dicts.
        Alpaca commonly sends a LIST of events per message; we flatten.
        """
        if not self._connected:
            await self.connect()

        await self.subscribe_bars(symbols)

        assert self._ws is not None
        while True:
            msg = await self._recv_json()

            # Alpaca sends list[dict]; keep tolerant
            if isinstance(msg, list):
                for ev in msg:
                    if isinstance(ev, dict):
                        ev["_received_ts_utc"] = _utc_now().isoformat()
                        yield ev
            elif isinstance(msg, dict):
                msg["_received_ts_utc"] = _utc_now().isoformat()
                yield msg
            else:
                # ignore unknown payloads
                continue
