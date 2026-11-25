"""Save / load MethodRun JSON under outputs/runs/<method>/<window_id>.json.

A sidecar (<window_id>.sidecar.json) holds method-specific extras: mined
patterns and DFGs for sequence mining, per-chunk usage for the LLM.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .schema import MethodRun, window_id_for


def run_path(runs_dir: str | Path, method: str, window_id: str) -> Path:
    return Path(runs_dir) / method / f"{window_id}.json"


def save_run(run: MethodRun, runs_dir: str | Path, sidecar: dict[str, Any] | None = None) -> Path:
    wid = window_id_for(run.window_start, run.window_end)
    path = run_path(runs_dir, run.method, wid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(run.model_dump_json(indent=2))
    if sidecar is not None:
        path.with_suffix(".sidecar.json").write_text(json.dumps(sidecar, indent=2, default=str))
    return path


def load_run(path: str | Path) -> MethodRun:
    return MethodRun.model_validate_json(Path(path).read_text())


def list_runs(runs_dir: str | Path, method: str) -> list[Path]:
    d = Path(runs_dir) / method
    return sorted(p for p in d.glob("*.json") if not p.name.endswith(".sidecar.json")) if d.exists() else []


def load_history(runs_dir: str | Path, method: str, before: datetime) -> list[dict[str, Any]]:
    """The method's OWN earlier results (windows ending at or before `before`), oldest first.

    sequence_mining -> per-task signatures from sidecars
    llm             -> task summaries and step descriptions
    """
    hist: list[dict[str, Any]] = []
    for p in list_runs(runs_dir, method):
        run = load_run(p)
        if run.window_end > before:
            continue
        if method == "sequence_mining":
            sc = p.with_suffix(".sidecar.json")
            if sc.exists():
                hist += json.loads(sc.read_text()).get("tasks", [])
        else:
            hist += [{"task_id": t.task_id, "label": t.label, "start": t.start.isoformat(),
                      "end": t.end.isoformat(), "summary": t.summary, "repeat_of": t.repeat_of,
                      "steps": [s.description for s in t.steps]} for t in run.tasks]
    return sorted(hist, key=lambda h: h["start"])
