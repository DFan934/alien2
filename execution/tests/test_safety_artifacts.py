# execution/tests/test_phase12_safety_artifacts.py
import asyncio
import json
from pathlib import Path

import pandas as pd
import pytest

from execution.contracts_live import OrderRequest, OrderUpdateEvent
from execution.core.contracts import SafetyAction
from execution.manager import ExecutionManager
from execution.brokers.base import BrokerAdapter
import inspect
import execution.manager
print("EXECUTION_MANAGER_FILE =", inspect.getsourcefile(execution.manager))


class DummyBroker(BrokerAdapter):
    async def stream_market_data(self, *args, **kwargs):
        raise NotImplementedError

    async def submit_order(self, req: OrderRequest) -> OrderUpdateEvent:
        # minimal ack
        return OrderUpdateEvent(
            client_order_id=req.client_order_id,
            broker_order_id="B1",
            symbol=req.symbol,
            side=req.side,
            status="ACK",
            filled_qty=0,
            fill_px=None,
            ts_utc=req.created_ts_utc,
        )

    async def stream_order_updates(self):
        # Attach-broker may spawn a consumer for this stream.
        # Provide an empty async generator so it doesn't crash.
        if False:
            yield None
        return



@pytest.mark.asyncio
async def test_step12_writes_policy_effective_json(tmp_path: Path):
    #mgr = ExecutionManager(config={"safety": {"cooldowns": {"DAILY_LOSS": 60}}}, log_path=tmp_path/"log.csv")

    mgr = ExecutionManager(equity=100_000)
    mgr.cfg["safety"] = {"cooldowns": {"DAILY_LOSS": 60}}

    mgr.attach_broker(DummyBroker(), out_dir=tmp_path)

    p = tmp_path / "safety_policy_effective.json"
    assert p.exists(), "safety_policy_effective.json must be written on attach_broker()"

    obj = json.loads(p.read_text(encoding="utf-8"))
    assert "rules" in obj
    assert any(r["trigger"] == "kill_switch" for r in obj["rules"])


@pytest.mark.asyncio
async def test_step12_kill_switch_blocks_and_logs_safety_action(tmp_path: Path):
    # engage latch
    (tmp_path / "KILL_SWITCH").write_text("ENGAGED\n", encoding="utf-8")

    #mgr = ExecutionManager(config={"live_order_limit_per_day": 10, "live_tiny_qty": 1}, log_path=tmp_path/"log.csv")

    mgr = ExecutionManager(equity=100_000)
    mgr.cfg["live_order_limit_per_day"] = 10
    mgr.cfg["live_tiny_qty"] = 1

    mgr.attach_broker(DummyBroker(), out_dir=tmp_path)

    watcher = asyncio.create_task(mgr._safety_watcher(), name="safety_watcher")
    try:
        # push a fake safety action too (force-trigger)
        mgr._safety_q.put_nowait(SafetyAction(action="HALT", reason="DAILY_LOSS"))

        # and attempt an order (should be blocked due to kill switch)
        req = OrderRequest(
            client_order_id="T1",
            symbol="AAPL",
            side="BUY",
            qty=5,
            order_type="MKT",
            tif="DAY",
        )
        evt = await mgr.submit_order_paper(req, reason="test")
        assert evt is None

        # give watcher time to flush parquet
        await asyncio.sleep(0.10)

        # attempted_actions.parquet should show blocked
        attempts_dir = tmp_path / "attempted_actions.parquet"
        parts = sorted(attempts_dir.glob("part-*.parquet"))
        assert parts, "attempted_actions.parquet must have part files"
        df_attempt = pd.read_parquet(parts[-1])
        assert df_attempt.iloc[-1]["allowed"] in (False, 0)
        assert df_attempt.iloc[-1]["block_reason"] in ("kill_switch", "safety_halt")

        # safety_actions.parquet should exist and contain at least one row
        sa_dir = tmp_path / "safety_actions.parquet"
        sa_parts = sorted(sa_dir.glob("part-*.parquet"))
        assert sa_parts, "safety_actions.parquet must be written"
        df_sa = pd.read_parquet(sa_parts[-1])
        assert len(df_sa) >= 1
        assert set(df_sa.columns) >= {"ts_utc", "action", "reason", "policy_action", "halt_active"}
    finally:
        watcher.cancel()
        try:
            await watcher
        except Exception:
            pass
