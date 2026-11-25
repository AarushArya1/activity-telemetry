"""Common output schema shared by every method (PROJECT_CONTEXT.md §7)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import AwareDatetime, BaseModel, Field


class Step(BaseModel):
    order: int
    description: str
    start: AwareDatetime
    end: AwareDatetime
    evidence_segment_ids: list[str]


class PredictedTask(BaseModel):
    task_id: str
    label: str
    start: AwareDatetime
    end: AwareDatetime
    summary: str
    steps: list[Step] = Field(default_factory=list)
    repeat_of: str | None = None
    confidence: float | None = None
    evidence_segment_ids: list[str] = Field(default_factory=list)


class MethodRun(BaseModel):
    method: str  # "sequence_mining" | "llm" | "baseline"
    method_version: str
    window_start: AwareDatetime
    window_end: AwareDatetime
    params: dict[str, Any] = Field(default_factory=dict)
    tasks: list[PredictedTask] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def window_id_for(start: datetime, end: datetime) -> str:
    """Filesystem-safe window id, e.g. '2026-11-23' for a whole local day."""
    if (end - start).total_seconds() == 86400 and start.hour == 0 and start.minute == 0:
        return start.strftime("%Y-%m-%d")
    return f"{start.strftime('%Y-%m-%dT%H%M')}_{end.strftime('%Y-%m-%dT%H%M')}"
