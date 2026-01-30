# execution/tests/test_phase14_unattended_gating.py
import json
from pathlib import Path

import pytest

from data_ingestion.live.gating import assert_unattended_allowed, write_promotion_readiness, LATCH_FILENAME


def test_unattended_blocked_without_latch(tmp_path: Path):
    write_promotion_readiness(tmp_path, {"gates_passed": True})
    with pytest.raises(RuntimeError):
        assert_unattended_allowed(tmp_path)


def test_unattended_blocked_when_gates_fail(tmp_path: Path):
    (tmp_path / LATCH_FILENAME).write_text("YES\n", encoding="utf-8")
    write_promotion_readiness(tmp_path, {"gates_passed": False})
    with pytest.raises(RuntimeError):
        assert_unattended_allowed(tmp_path)


def test_unattended_allowed_when_latch_and_gates_pass(tmp_path: Path):
    (tmp_path / LATCH_FILENAME).write_text("YES\n", encoding="utf-8")
    write_promotion_readiness(tmp_path, {"gates_passed": True})
    assert_unattended_allowed(tmp_path)
