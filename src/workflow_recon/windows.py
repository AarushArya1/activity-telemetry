"""Stage 2 — choose the context windows the methods see.

Default is a whole local day. Windows are never cut at logged task boundaries;
`padded_task` mode exists only as a debugging view (it leaks rough boundaries).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from .schema import window_id_for


@dataclass(frozen=True)
class Window:
    start: pd.Timestamp  # tz-aware, local tz
    end: pd.Timestamp
    mode: str

    @property
    def window_id(self) -> str:
        return window_id_for(self.start.to_pydatetime(), self.end.to_pydatetime())


def day_window(date: str | datetime | pd.Timestamp, tz: str) -> Window:
    day = pd.Timestamp(date)
    day = day.tz_convert(tz) if day.tzinfo else day.tz_localize(tz)
    start = day.normalize()
    end = (start.tz_localize(None) + pd.Timedelta(days=1)).tz_localize(tz)  # DST-safe midnight
    return Window(start, end, "day")


def available_days(events: pd.DataFrame, tz: str) -> list[pd.Timestamp]:
    """Local dates that have any non-AFK window activity."""
    act = events[(events["source"] == "window")]
    if act.empty:
        return []
    days = act["start_utc"].dt.tz_convert(tz).dt.normalize().unique()
    return sorted(pd.Timestamp(d) for d in days)


def day_windows(events: pd.DataFrame, tz: str) -> list[Window]:
    return [day_window(d, tz) for d in available_days(events, tz)]


def _active_intervals(events: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    afk = events[events["source"] == "afk"]
    src = afk[afk["status"] == "not-afk"] if not afk.empty else events[events["source"] == "window"]
    return list(zip(src["start_utc"], src["end_utc"]))


def session_windows(events: pd.DataFrame, tz: str, gap_min: float = 20) -> list[Window]:
    """Continuous activity between gaps of >= gap_min (sleep, long AFK, laptop off)."""
    ivs = sorted(_active_intervals(events))
    if not ivs:
        return []
    gap = pd.Timedelta(minutes=gap_min)
    out: list[Window] = []
    cs, ce = ivs[0]
    for s, e in ivs[1:]:
        if s - ce >= gap:
            out.append(Window(cs.tz_convert(tz), ce.tz_convert(tz), "session"))
            cs, ce = s, e
        else:
            ce = max(ce, e)
    out.append(Window(cs.tz_convert(tz), ce.tz_convert(tz), "session"))
    return out


def padded_task_windows(task_log: pd.DataFrame, padding_min: float = 20) -> list[Window]:
    """DEBUG ONLY: logged task windows ± padding, overlapping ones merged."""
    warnings.warn("padded_task windows leak approximate task boundaries; use only for debugging",
                  stacklevel=2)
    pad = pd.Timedelta(minutes=padding_min)
    ivs = sorted((r.start_local - pad, r.end_local + pad) for r in task_log.itertuples())
    out: list[Window] = []
    for s, e in ivs:
        if out and s <= out[-1].end:
            out[-1] = Window(out[-1].start, max(out[-1].end, e), "padded_task")
        else:
            out.append(Window(s, e, "padded_task"))
    return out


def slice_events(events: pd.DataFrame, window: Window) -> pd.DataFrame:
    """Events overlapping the window, clipped to its bounds."""
    ws, we = window.start.tz_convert("UTC"), window.end.tz_convert("UTC")
    m = (events["end_utc"] > ws) & (events["start_utc"] < we)
    out = events.loc[m].copy()
    out["start_utc"] = out["start_utc"].clip(lower=ws)
    out["end_utc"] = out["end_utc"].clip(upper=we)
    out["duration_s"] = (out["end_utc"] - out["start_utc"]).dt.total_seconds()
    tz = window.start.tz
    out["start_local"] = out["start_utc"].dt.tz_convert(tz)
    out["end_local"] = out["end_utc"].dt.tz_convert(tz)
    return out.reset_index(drop=True)


def select_windows(events: pd.DataFrame, cfg: dict, task_log: pd.DataFrame | None = None,
                   date: str | None = None) -> list[Window]:
    tz = cfg["timezone"]
    mode = cfg["window"]["mode"]
    if mode == "day":
        return [day_window(date, tz)] if date else day_windows(events, tz)
    if mode == "session":
        wins = session_windows(events, tz, cfg["window"]["session_gap_min"])
        if date:
            d = day_window(date, tz)
            wins = [w for w in wins if w.end > d.start and w.start < d.end]
        return wins
    if mode == "padded_task":
        if task_log is None:
            raise ValueError("padded_task mode needs the task log (debugging only)")
        return padded_task_windows(task_log, cfg["window"]["padding_min"])
    raise ValueError(f"unknown window mode {mode!r}")
