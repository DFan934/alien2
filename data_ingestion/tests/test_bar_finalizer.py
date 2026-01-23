from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from data_ingestion.live.bar_finalizer import BarFinalizer


def _ts_utc(y, m, d, hh, mm, ss=0):
    return datetime(y, m, d, hh, mm, ss, tzinfo=timezone.utc)


def test_finalizes_out_of_order_updates_in_correct_time_order():
    bf = BarFinalizer(freq="60s", emit_gaps=True)

    t_930 = _ts_utc(2026, 1, 22, 14, 30)  # 14:30 UTC
    t_931 = _ts_utc(2026, 1, 22, 14, 31)
    t_932 = _ts_utc(2026, 1, 22, 14, 32)

    # Ingest 09:31 first (no emit yet; watermark would be 09:30 but we don't have 09:30 buffered)
    out1 = bf.ingest_bar_update(symbol="AAPL", ts=t_931, open=1, high=2, low=0.5, close=1.5, volume=10)
    assert out1 == []

    # Then ingest 09:30 late -> should now emit 09:30 (watermark is still 09:30)
    out2 = bf.ingest_bar_update(symbol="AAPL", ts=t_930, open=10, high=12, low=9, close=11, volume=100)
    assert len(out2) == 1
    assert out2[0]["timestamp"] == pd.Timestamp(t_930)
    assert out2[0]["is_gap"] is False
    assert out2[0]["bar_present"] == 1
    assert out2[0]["symbol"] == "AAPL"

    # Ingest 09:32 -> watermark becomes 09:31, should emit the 09:31 bar
    out3 = bf.ingest_bar_update(symbol="AAPL", ts=t_932, open=2, high=3, low=1.5, close=2.5, volume=20)
    assert len(out3) == 1
    assert out3[0]["timestamp"] == pd.Timestamp(t_931)
    assert out3[0]["is_gap"] is False
    assert out3[0]["bar_present"] == 1
    assert out3[0]["symbol"] == "AAPL"


def test_emits_gap_rows_for_missing_minutes():
    bf = BarFinalizer(freq="60s", emit_gaps=True)

    t_930 = _ts_utc(2026, 1, 22, 14, 30)
    t_932 = _ts_utc(2026, 1, 22, 14, 32)  # missing 14:31

    # first bar -> no emit
    out1 = bf.ingest_bar_update(symbol="TSLA", ts=t_930, open=100, high=105, low=99, close=104, volume=1000)
    assert out1 == []

    # jump to 14:32 -> watermark becomes 14:31
    out2 = bf.ingest_bar_update(symbol="TSLA", ts=t_932, open=110, high=112, low=108, close=111, volume=2000)

    # We should emit 14:30 (present) and 14:31 (gap)
    assert len(out2) == 2
    assert out2[0]["timestamp"] == pd.Timestamp(t_930)
    assert out2[0]["is_gap"] is False
    assert out2[0]["bar_present"] == 1

    assert out2[1]["timestamp"] == pd.Timestamp(_ts_utc(2026, 1, 22, 14, 31))
    assert out2[1]["is_gap"] is True
    assert out2[1]["bar_present"] == 0
    assert pd.isna(out2[1]["open"])
    assert pd.isna(out2[1]["close"])
    assert out2[1]["volume"] == 0


def test_normalizes_to_utc_and_floors_to_minute():
    bf = BarFinalizer(freq="60s", emit_gaps=False)

    # A tz-aware non-UTC timestamp: UTC-05:00 (fixed offset for test simplicity)
    tz_minus5 = timezone(timedelta(hours=-5))
    local = datetime(2026, 1, 22, 9, 30, 42, tzinfo=tz_minus5)  # corresponds to 14:30:42 UTC

    out1 = bf.ingest_bar_update(symbol="MSFT", ts=local, open=1, high=1, low=1, close=1, volume=1)
    assert out1 == []

    # Next minute arrives so first one finalizes
    local2 = datetime(2026, 1, 22, 9, 31, 1, tzinfo=tz_minus5)  # 14:31:01 UTC
    out2 = bf.ingest_bar_update(symbol="MSFT", ts=local2, open=2, high=2, low=2, close=2, volume=2)

    assert len(out2) == 1
    # Floored to 14:30:00 UTC
    assert out2[0]["timestamp"] == pd.Timestamp(datetime(2026, 1, 22, 14, 30, 0, tzinfo=timezone.utc))
    assert str(out2[0]["timestamp"].tz) in ("UTC", "UTC+00:00", "UTC+0000")


def test_parquet_roundtrip_schema(tmp_path):
    bf = BarFinalizer(freq="60s", emit_gaps=True)

    t_930 = _ts_utc(2026, 1, 22, 14, 30)
    t_932 = _ts_utc(2026, 1, 22, 14, 32)

    bf.ingest_bar_update(symbol="NVDA", ts=t_930, open=10, high=11, low=9, close=10.5, volume=500)
    out = bf.ingest_bar_update(symbol="NVDA", ts=t_932, open=12, high=13, low=11, close=12.5, volume=600)

    df = pd.DataFrame(out)
    # Must have canonical columns + live extras
    for c in ["timestamp", "symbol", "open", "high", "low", "close", "volume", "bar_present", "is_gap"]:
        assert c in df.columns

    # timestamp must be tz-aware UTC
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
    assert df["timestamp"].dt.tz is not None

    fp = tmp_path / "bars.parquet"
    df.to_parquet(fp, index=False)
    df2 = pd.read_parquet(fp)

    # Basic schema check after round-trip
    for c in ["timestamp", "symbol", "open", "high", "low", "close", "volume", "bar_present", "is_gap"]:
        assert c in df2.columns
