# scripts/live_broker_smoke.py
from __future__ import annotations

import asyncio
from pathlib import Path

from data_ingestion.utils import load_config
from execution.brokers.alpaca_paper import AlpacaPaperAdapter
from execution.brokers.base import BrokerCallPolicy

# IMPORTANT:
# Replace the import below to match the module where your RunManifest helpers actually live.
# In your snapshot, these helpers exist (write_run_manifest / update_run_manifest_fields),
# but the exact module path is your project's manifest module. :contentReference[oaicite:1]{index=1}
from feature_engineering.utils.artifacts_root import write_run_manifest, update_run_manifest_fields


def _mk_policy(cfg: dict) -> BrokerCallPolicy:
    pol_cfg = ((cfg.get("broker") or {}).get("call_policy") or {})
    # Use YOUR base.py policy fields (BrokerCallPolicy), not the old CallPolicy.
    return BrokerCallPolicy(
        timeout_s=float(pol_cfg.get("timeout_s", BrokerCallPolicy().timeout_s)),
        max_retries=int(pol_cfg.get("max_retries", BrokerCallPolicy().max_retries)),
        backoff_s=tuple(pol_cfg.get("backoff_s", BrokerCallPolicy().backoff_s)),
        circuit_breaker_failures=int(
            pol_cfg.get("circuit_breaker_failures", BrokerCallPolicy().circuit_breaker_failures)
        ),
        circuit_breaker_cooloff_s=float(
            pol_cfg.get("circuit_breaker_cooloff_s", BrokerCallPolicy().circuit_breaker_cooloff_s)
        ),
    )


async def main() -> None:
    cfg = load_config()

    # Your codebase already has canonical artifacts_root + run_dir behavior via RunContext/manifest.
    # Use the same run_dir that run_backtest/live scripts already use.
    # Minimal safe assumption: cfg contains artifacts_root, and your manifest helpers accept run_dir.
    artifacts_root = Path(cfg.get("artifacts_root", "artifacts"))
    run_dir = artifacts_root / (cfg.get("run_id") or "live_smoke")
    run_dir.mkdir(parents=True, exist_ok=True)

    broker = AlpacaPaperAdapter(cfg)
    await broker.connect()

    # prove broker endpoints work
    clk = await broker.get_clock()
    acct = await broker.get_account()

    print("[BrokerSmoke] clock_is_open=", clk.is_open, "ts_utc=", clk.ts_utc)
    print("[BrokerSmoke] account_id=", acct.id, "status=", acct.status, "equity=", acct.equity)

    # Write/Update manifest with broker fields (after your RunManifest patch)
    write_run_manifest(
        run_dir=run_dir,
        run_id=str(cfg.get("run_id") or "live_smoke"),
        artifacts_root=artifacts_root,
        config_hash=str(cfg.get("config_hash") or cfg.get("_config_hash") or ""),
        clock_hash=str(cfg.get("_unified_clock_hash") or ""),
        broker_name="alpaca_paper",
        broker_paper=True,
    )

    update_run_manifest_fields(
        run_dir,
        broker_name="alpaca_paper",
        broker_paper=True,
    )

    await broker.close()


if __name__ == "__main__":
    asyncio.run(main())
