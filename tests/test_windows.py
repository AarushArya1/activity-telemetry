import warnings

import pandas as pd
import pytest

from workflow_recon.windows import (available_days, day_window, padded_task_windows, select_windows,
                                    session_windows, slice_events)

TZ = "America/Los_Angeles"


def test_day_window_is_local_midnight():
    w = day_window("2026-12-01", TZ)
    assert w.start == pd.Timestamp("2026-12-01 00:00", tz=TZ)
    assert (w.end - w.start) == pd.Timedelta(hours=24)
    assert w.window_id == "2026-12-01"


def test_day_window_dst_safe():
    w = day_window("2026-11-01", TZ)  # DST ends: 25-hour day
    assert w.end == pd.Timestamp("2026-11-02 00:00", tz=TZ)


def test_available_days(events):
    assert [d.strftime("%Y-%m-%d") for d in available_days(events, TZ)] == ["2026-12-01", "2026-12-02"]


def test_sessions_split_on_long_gaps(events):
    wins = session_windows(events, TZ, gap_min=20)
    assert len(wins) >= 4
    for a, b in zip(wins, wins[1:]):
        assert (b.start - a.end) >= pd.Timedelta(minutes=20)


def test_slice_clips_to_window(events):
    w = day_window("2026-12-01", TZ)
    s = slice_events(events, w)
    assert (s["start_utc"] >= w.start.tz_convert("UTC")).all()
    assert (s["end_utc"] <= w.end.tz_convert("UTC")).all()


def test_padded_task_mode_warns_and_needs_log(events, fake_log, cfg):
    cfg["window"]["mode"] = "padded_task"
    with pytest.raises(ValueError):
        select_windows(events, cfg)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        wins = padded_task_windows(fake_log, 20)
    assert rec and wins


def test_default_mode_is_day(cfg):
    assert cfg["window"]["mode"] == "day"
