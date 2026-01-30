# execution/safety_policy.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from execution.core.contracts import SafetyAction
from execution.safety import HaltReason


@dataclass(frozen=True)
class PolicyRule:
    """
    Formal mapping: trigger -> action -> reset.
    This is the policy layer *on top of* SafetyFSM signals.
    """
    trigger: str                 # e.g. "DAILY_LOSS", "MICRO_HALT", "kill_switch"
    action: str                  # "HALT" | "RESUME" | "SIZE_DOWN"
    reset_seconds: Optional[int] # None => manual/implicit reset


@dataclass(frozen=True)
class PolicyDecision:
    trigger: str
    action: str
    applied_ts_utc: datetime
    reset_ts_utc: Optional[datetime]
    note: Optional[str] = None


class SafetyPolicy:
    """
    Step 12:
      - Formalizes policy mapping trigger -> action -> reset
      - Produces "effective" policy JSON for artifacts
      - Helps ExecutionManager apply consistent effects (halt / size-down)
    """

    def __init__(self, rules: Dict[str, PolicyRule]) -> None:
        self._rules = rules

    @staticmethod
    def from_config(cfg: Optional[dict], *, safety_cfg: Optional[dict] = None) -> "SafetyPolicy":
        """
        Builds rules from:
          - cfg["rules"] (explicit)
          - otherwise, defaults derived from safety_cfg["cooldowns"] keyed by HaltReason
        """
        cfg = cfg or {}
        safety_cfg = safety_cfg or {}

        rules: Dict[str, PolicyRule] = {}

        # 1) Explicit rules (if present)
        for r in (cfg.get("rules") or []):
            trigger = str(r["trigger"])
            rules[trigger] = PolicyRule(
                trigger=trigger,
                action=str(r.get("action", "HALT")),
                reset_seconds=(None if r.get("reset_seconds") is None else int(r["reset_seconds"])),
            )

        # 2) Defaults from SafetyFSM cooldowns if not explicitly provided
        cooldowns = safety_cfg.get("cooldowns") or {}
        for reason in HaltReason:
            key = reason.name
            if key in rules:
                continue
            reset_s = cooldowns.get(key)
            rules[key] = PolicyRule(
                trigger=key,
                action="HALT",
                reset_seconds=(None if reset_s is None else int(reset_s)),
            )

        # 3) Add kill switch default (manual reset)
        if "kill_switch" not in rules:
            rules["kill_switch"] = PolicyRule(
                trigger="kill_switch",
                action="HALT",
                reset_seconds=None,
            )

        # 4) Profit recovery / resume (if SafetyFSM emits it)
        if "profit_recovery" not in rules:
            rules["profit_recovery"] = PolicyRule(
                trigger="profit_recovery",
                action="RESUME",
                reset_seconds=None,
            )

        return SafetyPolicy(rules)

    def to_effective_dict(self) -> dict:
        return {
            "rules": [
                {
                    "trigger": r.trigger,
                    "action": r.action,
                    "reset_seconds": r.reset_seconds,
                }
                for r in self._rules.values()
            ]
        }

    def decide(self, safety_action: SafetyAction, *, now: Optional[datetime] = None) -> PolicyDecision:
        now = now or datetime.now(timezone.utc)
        trigger = (safety_action.reason or "").strip() or "unknown"
        rule = self._rules.get(trigger)

        # If SafetyFSM uses HaltReason names, we’ll match them directly.
        # If it emits other reasons (e.g. "profit_recovery"), we also handle.
        if rule is None:
            # Unknown triggers default to HALT with no reset.
            return PolicyDecision(
                trigger=trigger,
                action=safety_action.action,
                applied_ts_utc=now,
                reset_ts_utc=None,
                note="no_rule_default",
            )

        reset_ts = None
        if rule.reset_seconds is not None and rule.action == "HALT":
            reset_ts = now + timedelta(seconds=int(rule.reset_seconds))

        return PolicyDecision(
            trigger=rule.trigger,
            action=rule.action,
            applied_ts_utc=now,
            reset_ts_utc=reset_ts,
            note=None,
        )
