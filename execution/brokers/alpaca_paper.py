from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Optional

import urllib.request
import urllib.error

from execution.brokers.base import (
    BrokerAdapter,
    BrokerCallPolicy,
    CircuitBreaker,
    BrokerError,
    BrokerAuthError,
    BrokerRateLimited,
    BrokerRejected,
    BrokerUnavailable,
    BrokerDisconnected,
    BrokerInvalidRequest,
)
from execution.contracts_live import BrokerClock, OrderRequest, OrderUpdateEvent
from data_ingestion.contracts import MarketDataEvent


def _env_expand(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if s.startswith("${") and s.endswith("}"):
        key = s[2:-1].strip()
        return os.environ.get(key)
    return s


def _parse_rfc3339_to_utc(s: str) -> datetime:
    """
    Alpaca clock fields come as RFC3339, often with 'Z'.
    Convert to tz-aware UTC datetime.
    """
    if not isinstance(s, str) or not s:
        raise ValueError(f"bad timestamp: {s!r}")
    # Handle trailing Z
    s2 = s.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s2)
    if dt.tzinfo is None:
        raise ValueError("broker returned naive timestamp")
    return dt.astimezone(timezone.utc)


@dataclass(frozen=True)
class AlpacaAccount:
    id: str
    status: str
    currency: str
    equity: float
    buying_power: float
    cash: float

    @staticmethod
    def from_api(d: Dict[str, Any]) -> "AlpacaAccount":
        def f(x: Any) -> float:
            try:
                return float(x)
            except Exception:
                return 0.0

        return AlpacaAccount(
            id=str(d.get("id", "")),
            status=str(d.get("status", "")),
            currency=str(d.get("currency", "")),
            equity=f(d.get("equity")),
            buying_power=f(d.get("buying_power")),
            cash=f(d.get("cash")),
        )


class AlpacaPaperAdapter:
    """
    Task 4 scope:
      - connect/close
      - get_clock
      - get_account

    Does NOT implement market-data streaming or order submission yet.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        broker_cfg = (cfg.get("broker") or {})
        alpaca_cfg = (broker_cfg.get("alpaca") or {})

        self._key_id = _env_expand(alpaca_cfg.get("key_id"))
        self._secret_key = _env_expand(alpaca_cfg.get("secret_key"))

        self._base_url = str(alpaca_cfg.get("trading_base_url") or "https://paper-api.alpaca.markets").rstrip("/")
        self._api_version = str(alpaca_cfg.get("api_version") or "v2").strip()

        # Policy defaults should align with your repo defaults:
        # BrokerCallPolicy(timeout_s=5, max_retries=3, backoff_s=(1,2,5), breaker_failures=3, breaker_cooloff_s=60)
        pol_cfg = (broker_cfg.get("call_policy") or {})
        self.policy = BrokerCallPolicy(
            timeout_s=float(pol_cfg.get("timeout_s", BrokerCallPolicy().timeout_s)),
            max_retries=int(pol_cfg.get("max_retries", BrokerCallPolicy().max_retries)),
            backoff_s=tuple(pol_cfg.get("backoff_s", BrokerCallPolicy().backoff_s)),
            circuit_breaker_failures=int(pol_cfg.get("circuit_breaker_failures", BrokerCallPolicy().circuit_breaker_failures)),
            circuit_breaker_cooloff_s=float(pol_cfg.get("circuit_breaker_cooloff_s", BrokerCallPolicy().circuit_breaker_cooloff_s)),
        )

        if not self._key_id or not self._secret_key:
            raise BrokerAuthError(
                "Missing Alpaca credentials. Set broker.alpaca.key_id/secret_key "
                "(supports ${ENV_VAR}) and ensure env vars exist."
            )

        self._connected = False
        self._breaker = CircuitBreaker(
            fail_threshold=self.policy.circuit_breaker_failures,
            cooloff_s=self.policy.circuit_breaker_cooloff_s,
        )

    async def connect(self) -> None:
        self._connected = True

    async def close(self) -> None:
        self._connected = False

    # ----------------------------
    # Task 4 methods
    # ----------------------------

    async def get_clock(self) -> BrokerClock:
        payload = await self._call_json("GET", f"/{self._api_version}/clock", None)

        ts = _parse_rfc3339_to_utc(payload["timestamp"])
        next_open = _parse_rfc3339_to_utc(payload["next_open"]) if payload.get("next_open") else None
        next_close = _parse_rfc3339_to_utc(payload["next_close"]) if payload.get("next_close") else None

        return BrokerClock(
            ts_utc=ts,
            is_open=bool(payload.get("is_open", False)),
            next_open_utc=next_open,
            next_close_utc=next_close,
            raw={"alpaca": True},
        )

    async def get_account(self) -> AlpacaAccount:
        payload = await self._call_json("GET", f"/{self._api_version}/account", None)
        return AlpacaAccount.from_api(payload)

    # ----------------------------
    # Not in Task 4 scope
    # ----------------------------

    async def stream_market_data(self, *, symbols: list[str]) -> AsyncIterator[MarketDataEvent]:
        raise NotImplementedError("Task 5+ will add Alpaca market data streaming")

    async def submit_order(self, req: OrderRequest) -> OrderUpdateEvent:
        raise NotImplementedError("Task 6+ will add Alpaca order submission")

    # ----------------------------
    # Internal HTTP + retry/breaker
    # ----------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": str(self._key_id),
            "APCA-API-SECRET-KEY": str(self._secret_key),
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    async def _call_json(self, method: str, path: str, body: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not self._connected:
            raise BrokerDisconnected("alpaca adapter not connected")

        if self._breaker.is_open():
            raise BrokerUnavailable("circuit breaker open")

        attempts = 1 + int(self.policy.max_retries)
        backoffs = list(self.policy.backoff_s)
        if not backoffs:
            backoffs = [1.0]

        last_err: Optional[Exception] = None

        for i in range(attempts):
            try:
                out = await self._http_json_once(method, path, body)
                self._breaker.record_success()
                return out
            except BrokerRateLimited as e:
                last_err = e
                self._breaker.record_failure()
                # use provided retry_after if meaningful else fallback
                sleep_s = float(e.retry_after_s) if getattr(e, "retry_after_s", 0.0) > 0 else float(backoffs[min(i, len(backoffs)-1)])
                await asyncio.sleep(sleep_s)
            except (BrokerUnavailable, urllib.error.URLError) as e:
                last_err = e
                self._breaker.record_failure()
                if i < attempts - 1:
                    await asyncio.sleep(float(backoffs[min(i, len(backoffs)-1)]))
            except BrokerError as e:
                # non-retryable (auth/rejected/invalid)
                self._breaker.record_failure()
                raise

        raise BrokerUnavailable(f"alpaca call failed after {attempts} attempts: {last_err}")

    async def _http_json_once(self, method: str, path: str, body: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        url = f"{self._base_url}{path}"
        data = None if body is None else json.dumps(body).encode("utf-8")

        def _do() -> Dict[str, Any]:
            req = urllib.request.Request(url=url, data=data, method=method.upper(), headers=self._headers())
            try:
                with urllib.request.urlopen(req, timeout=float(self.policy.timeout_s)) as resp:
                    raw = resp.read().decode("utf-8")
                    return json.loads(raw) if raw else {}
            except urllib.error.HTTPError as e:
                raw = ""
                try:
                    raw = e.read().decode("utf-8")
                except Exception:
                    pass

                code = int(getattr(e, "code", 0) or 0)

                if code in (401, 403):
                    raise BrokerAuthError(f"alpaca auth failed ({code}): {raw}")

                if code == 429:
                    # Retry-After is optional; BrokerRateLimited expects retry_after_s
                    ra = 0.0
                    try:
                        h = e.headers.get("Retry-After")
                        if h:
                            ra = float(h)
                    except Exception:
                        ra = 0.0
                    raise BrokerRateLimited(retry_after_s=ra, message=f"alpaca rate limited ({code}): {raw}")

                if 500 <= code <= 599:
                    raise BrokerUnavailable(f"alpaca server error ({code}): {raw}")

                # other 4xx -> reject
                raise BrokerRejected(reason=f"alpaca HTTP error ({code}): {raw}", code=str(code))

            except urllib.error.URLError as e:
                raise BrokerUnavailable(f"network error calling alpaca: {e}")

        return await asyncio.to_thread(_do)
