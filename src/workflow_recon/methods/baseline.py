"""Baseline — label each segment by the same rules as Method A, no smoothing,
segmentation or steps. Consecutive segments with the same label form one task."""

from __future__ import annotations

import pandas as pd

from ..schema import MethodRun, PredictedTask
from ..windows import Window
from .common import add_label_hints, runs

METHOD = "baseline"
METHOD_VERSION = "base-v1"


def run_baseline(segments: pd.DataFrame, window: Window, cfg: dict, max_gap_s: float = 300) -> MethodRun:
    run = MethodRun(method=METHOD, method_version=METHOD_VERSION,
                    window_start=window.start.to_pydatetime(), window_end=window.end.to_pydatetime(),
                    params={"max_gap_s": max_gap_s})
    if segments.empty:
        run.warnings.append("no segments in window")
        return run
    df = add_label_hints(segments).sort_values("start").reset_index(drop=True)
    # break runs on idle gaps too
    gap_break = [0]
    for i in range(1, len(df)):
        gap = (df.at[i, "start"] - df.at[i - 1, "end"]).total_seconds()
        gap_break.append(gap_break[-1] + int(gap >= max_gap_s))
    keys = list(zip(gap_break, df["hint_label"]))
    for n, (a, b) in enumerate(runs(keys), 1):
        sub = df.iloc[a:b]
        label = sub["hint_label"].iloc[0]
        dom = sub.groupby("token")["duration_s"].sum().idxmax()
        run.tasks.append(PredictedTask(
            task_id=f"BL-{window.window_id}-{n:03d}", label=label,
            start=sub["start"].iloc[0].floor("us").to_pydatetime(), end=sub["end"].max().floor("us").to_pydatetime(),
            summary=f"{label} (dominant: {dom})", steps=[],
            confidence=round(float(sub["hint_strong"].mean()), 3),
            evidence_segment_ids=list(sub["segment_id"])))
    return run
