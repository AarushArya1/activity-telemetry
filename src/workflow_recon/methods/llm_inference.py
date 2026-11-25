"""Method B — LLM workflow inference with grounded, validated citations.

1. Render the REDACTED segment table as compact text lines (one per segment).
2. Chunk long windows (prefer splitting at long idle gaps, small overlap otherwise).
3. Ask the model for structured JSON (tasks, steps, cited segment ids).
4. Validate: drop citations to nonexistent / out-of-range segments, drop steps left
   with no evidence (counted as hallucinated), fix invalid labels / repeat refs.
5. Merge tasks that span chunk boundaries; assign final ids; resolve repeat_of.

Privacy: refuses to run on a table that has not been through redact.py, and runs
a last regex leak check on the exact text before it is sent.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ValidationError

from ..config import REPO_ROOT
from ..labels import LOTHER, PREDICTABLE_LABELS, taxonomy_text
from ..llm import LLMError, LLMProvider, provider_settings
from ..redact import find_leaks
from ..schema import MethodRun, PredictedTask, Step
from ..windows import Window

METHOD = "llm"
METHOD_VERSION = "b-v1"


class PrivacyError(RuntimeError):
    pass


# ------------------------------------------------------------ output schema (model-facing)

class _StepOut(BaseModel):
    description: str
    evidence_segment_ids: list[str]


class _TaskOut(BaseModel):
    local_id: str
    label: str
    start_segment_id: str
    end_segment_id: str
    summary: str
    steps: list[_StepOut]
    repeat_of: str | None
    confidence: float


class _Out(BaseModel):
    tasks: list[_TaskOut]


def output_json_schema() -> dict[str, Any]:
    step = {"type": "object", "additionalProperties": False,
            "required": ["description", "evidence_segment_ids"],
            "properties": {"description": {"type": "string"},
                           "evidence_segment_ids": {"type": "array", "items": {"type": "string"}}}}
    task = {"type": "object", "additionalProperties": False,
            "required": ["local_id", "label", "start_segment_id", "end_segment_id", "summary", "steps",
                         "repeat_of", "confidence"],
            "properties": {"local_id": {"type": "string"},
                           "label": {"type": "string", "enum": PREDICTABLE_LABELS},
                           "start_segment_id": {"type": "string"},
                           "end_segment_id": {"type": "string"},
                           "summary": {"type": "string"},
                           "steps": {"type": "array", "items": step},
                           "repeat_of": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                           "confidence": {"type": "number"}}}
    return {"type": "object", "additionalProperties": False, "required": ["tasks"],
            "properties": {"tasks": {"type": "array", "items": task}}}


# ------------------------------------------------------------ prompt

def load_prompt(version: str, prompts_dir: Path | None = None) -> tuple[str, str]:
    path = (prompts_dir or REPO_ROOT / "prompts") / f"{version}_workflow_inference.txt"
    text = path.read_text()
    _, rest = text.split("=== SYSTEM ===", 1)
    system, user = rest.split("=== USER ===", 1)
    return system.strip(), user.strip()


def _fmt_dur(s: float) -> str:
    s = int(round(s))
    return f"{s // 3600}h{(s % 3600) // 60:02d}m" if s >= 3600 else (f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s")


def render_segments(segments: pd.DataFrame, idle_min_s: float = 120) -> str:
    lines = []
    prev_end = None
    for r in segments.itertuples(index=False):
        if prev_end is not None and (r.start - prev_end).total_seconds() >= idle_min_s:
            lines.append(f"--- idle {prev_end:%H:%M}–{r.start:%H:%M} "
                         f"({_fmt_dur((r.start - prev_end).total_seconds())}) ---")
        where = r.domain or r.app or "unknown"
        parts = [f"[{r.segment_id}] {r.start:%H:%M:%S}–{r.end:%H:%M:%S} ({_fmt_dur(r.duration_s)})", where,
                 r.title_norm or ""]
        url = r.url_norm or ""
        if url and url != r.domain:
            parts.append(url)
        if isinstance(r.detail, str) and r.detail:
            parts.append(r.detail)
        lines.append(" | ".join(p for p in parts if p is not None))
        prev_end = r.end if prev_end is None else max(prev_end, r.end)
    return "\n".join(lines)


def render_history(history: list[dict]) -> str:
    if not history:
        return "(none)"
    out = []
    for h in history:
        steps = "; ".join(h.get("steps", []))[:300]
        out.append(f"- {h['task_id']} | {h['start'][:16]} | {h['label']} | {h['summary']}"
                   + (f" | steps: {steps}" if steps else ""))
    return "\n".join(out)


def build_user_prompt(template: str, window: Window, tz: str, segments_text: str, history_text: str) -> str:
    return (template.replace("{{window}}", f"{window.start:%Y-%m-%d %H:%M} to {window.end:%Y-%m-%d %H:%M}")
            .replace("{{timezone}}", tz).replace("{{history}}", history_text)
            .replace("{{segments}}", segments_text))


# ------------------------------------------------------------ chunking

def chunk_segments(segments: pd.DataFrame, max_n: int, overlap: int, gap_min: float) -> list[pd.DataFrame]:
    n = len(segments)
    if n <= max_n:
        return [segments]
    starts = list(segments["start"])
    ends = list(segments["end"])
    chunks, a = [], 0
    while a < n:
        b = min(n, a + max_n)
        if b == n:
            chunks.append(segments.iloc[a:b])
            break
        # best split: largest gap in the last 30% of the chunk
        lo = a + int(0.7 * (b - a))
        gaps = [((starts[i] - ends[i - 1]).total_seconds(), i) for i in range(max(lo, a + 1), b)]
        gap, cut = max(gaps) if gaps else (0, b)
        chunks.append(segments.iloc[a:cut])
        a = cut if gap >= gap_min * 60 else max(cut - overlap, a + 1)
    return chunks


# ------------------------------------------------------------ validation

def validate_chunk(out: _Out, chunk: pd.DataFrame, chunk_idx: int, stats: dict, warnings: list[str]) -> list[dict]:
    pos = {sid: i for i, sid in enumerate(chunk["segment_id"])}
    starts = list(chunk["start"])
    ends = list(chunk["end"])
    ids = list(chunk["segment_id"])
    tasks = []
    for t in out.tasks:
        label = t.label
        if label not in PREDICTABLE_LABELS:
            warnings.append(f"chunk {chunk_idx} {t.local_id}: invalid label {label!r} -> Other")
            stats["invalid_labels"] += 1
            label = LOTHER
        cited = [sid for s in t.steps for sid in s.evidence_segment_ids if sid in pos]
        ps, pe = pos.get(t.start_segment_id), pos.get(t.end_segment_id)
        if ps is None or pe is None:
            stats["invalid_task_bounds"] += 1
            if not cited:
                warnings.append(f"chunk {chunk_idx} {t.local_id}: no valid bounds or citations; task dropped")
                stats["tasks_dropped"] += 1
                stats["hallucinated_citations"] += sum(len(s.evidence_segment_ids) for s in t.steps)
                stats["hallucinated_steps"] += len(t.steps)
                stats["steps_total"] += len(t.steps)
                continue
            ps = min(pos[c] for c in cited) if ps is None else ps
            pe = max(pos[c] for c in cited) if pe is None else pe
        if ps > pe:
            ps, pe = pe, ps
        steps = []
        for s in t.steps:
            good = []
            for sid in s.evidence_segment_ids:
                if sid not in pos:
                    stats["hallucinated_citations"] += 1
                elif not ps <= pos[sid] <= pe:
                    stats["out_of_range_citations"] += 1
                else:
                    good.append(sid)
            good = sorted(set(good), key=pos.get)
            if not good:
                stats["hallucinated_steps"] += 1
                continue
            steps.append({"description": s.description, "evidence": good,
                          "start": starts[pos[good[0]]], "end": max(ends[pos[g]] for g in good)})
        stats["steps_total"] += len(t.steps)
        tasks.append({"chunk": chunk_idx, "local_id": t.local_id, "label": label,
                      "start": starts[ps], "end": max(ends[ps:pe + 1]),
                      "summary": t.summary, "steps": steps, "repeat_of": t.repeat_of,
                      "confidence": max(0.0, min(1.0, float(t.confidence))),
                      "evidence": ids[ps:pe + 1]})
    return tasks


def merge_boundary_tasks(tasks: list[dict], gap_s: float) -> list[dict]:
    """Merge same-label tasks from different chunks that touch or overlap."""
    tasks = sorted(tasks, key=lambda t: t["start"])
    out: list[dict] = []
    for t in tasks:
        if out:
            p = out[-1]
            if (p["chunk"] != t["chunk"] and p["label"] == t["label"]
                    and (t["start"] - p["end"]).total_seconds() <= gap_s):
                seen = set(p["evidence"])
                p["steps"] += [s for s in t["steps"] if not set(s["evidence"]) <= seen]
                p["evidence"] = list(dict.fromkeys(p["evidence"] + t["evidence"]))
                p["end"] = max(p["end"], t["end"])
                p["aliases"] = p.get("aliases", []) + [(t["chunk"], t["local_id"])]
                p["confidence"] = min(p["confidence"], t["confidence"])
                continue
        out.append(t)
    return out


# ------------------------------------------------------------ main

def run_llm(segments: pd.DataFrame, window: Window, cfg: dict, provider: LLMProvider,
            history: list[dict] | None = None) -> tuple[MethodRun, dict[str, Any]]:
    if not segments.attrs.get("redacted", False):
        raise PrivacyError("run_llm requires a redacted segment table (run redact.redact_segments first)")
    t0 = time.time()
    lc = cfg["llm"]
    system_t, user_t = load_prompt(lc["prompt_version"])
    system = system_t.replace("{{taxonomy}}", taxonomy_text())
    schema = output_json_schema()
    history = list(history or [])[-lc["history_tasks"]:]
    max_tokens = int(provider_settings(lc).get("max_tokens", 32000))
    tz = cfg["timezone"]

    run = MethodRun(method=METHOD, method_version=METHOD_VERSION,
                    window_start=window.start.to_pydatetime(), window_end=window.end.to_pydatetime(),
                    params={"llm": {**{k: v for k, v in lc.items() if not isinstance(v, dict)},
                                    **provider_settings(lc), **provider.settings()},
                            "prompt_sha256": hashlib.sha256((system_t + user_t).encode()).hexdigest()[:16],
                            "clean": cfg["clean"], "window": cfg["window"]})
    sidecar: dict[str, Any] = {"chunks": [], "usage": {}}
    if segments.empty:
        run.warnings.append("no segments in window")
        return run, sidecar

    segments = segments.sort_values("start").reset_index(drop=True)
    chunks = chunk_segments(segments, lc["max_segments_per_chunk"], lc["chunk_overlap_segments"], lc["chunk_gap_min"])
    stats = {k: 0 for k in ("hallucinated_citations", "out_of_range_citations", "hallucinated_steps",
                            "steps_total", "invalid_labels", "invalid_task_bounds", "tasks_dropped",
                            "invalid_repeat_refs", "chunks_failed")}
    usage_tot: dict[str, int] = {}
    all_tasks: list[dict] = []
    running_history = list(history)
    for ci, chunk in enumerate(chunks):
        user = build_user_prompt(user_t, window, tz, render_segments(chunk), render_history(running_history))
        leaks = find_leaks(system + "\n" + user)
        if leaks:
            raise PrivacyError(f"unredacted entities in LLM input: {leaks}; refusing to send")
        try:
            resp = provider.generate_json(system, user, schema, max_tokens)
            out = _Out.model_validate(resp.data)
        except (LLMError, ValidationError) as e:
            stats["chunks_failed"] += 1
            run.warnings.append(f"chunk {ci} failed: {type(e).__name__}: {str(e)[:200]}")
            continue
        for k, v in resp.usage.items():
            usage_tot[k] = usage_tot.get(k, 0) + v
        sidecar["chunks"].append({"index": ci, "segments": len(chunk), "first": chunk["segment_id"].iloc[0],
                                  "last": chunk["segment_id"].iloc[-1], "cached": resp.cached,
                                  "model": resp.model, "usage": resp.usage, "tasks": len(out.tasks)})
        tasks = validate_chunk(out, chunk, ci, stats, run.warnings)
        all_tasks += tasks
        # let later chunks see this chunk's tasks for repeat detection
        running_history += [{"task_id": f"c{ci}:{t['local_id']}", "label": t["label"],
                             "start": t["start"].isoformat(), "summary": t["summary"],
                             "steps": [s["description"] for s in t["steps"]]} for t in tasks]

    merged = merge_boundary_tasks(all_tasks, lc["boundary_merge_gap_s"])
    final = _finalize(merged, window, history, stats, run.warnings)
    run.tasks = final
    if stats["hallucinated_citations"] or stats["hallucinated_steps"] or stats["out_of_range_citations"]:
        run.warnings.append(
            f"dropped {stats['hallucinated_citations']} nonexistent and {stats['out_of_range_citations']} "
            f"out-of-range citations; {stats['hallucinated_steps']} of {stats['steps_total']} steps had no valid evidence")
    run.params["stats"] = {**stats, "chunks": len(chunks), "segments": len(segments),
                           "runtime_s": round(time.time() - t0, 2)}
    run.params["usage"] = usage_tot
    sidecar["usage"] = usage_tot
    return run, sidecar


def _finalize(merged: list[dict], window: Window, history: list[dict], stats: dict, warnings: list[str]) -> list[PredictedTask]:
    wid = window.window_id
    id_map: dict[tuple, str] = {}
    for n, t in enumerate(merged, 1):
        t["task_id"] = f"B-{wid}-{n:03d}"
        id_map[(t["chunk"], t["local_id"])] = t["task_id"]
        id_map[("hist", f"c{t['chunk']}:{t['local_id']}")] = t["task_id"]
        for alias in t.get("aliases", []):
            id_map[alias] = t["task_id"]
    hist_root = {h["task_id"]: h.get("repeat_of") or h["task_id"] for h in history}
    order = {t["task_id"]: i for i, t in enumerate(merged)}
    root: dict[str, str] = dict(hist_root)
    out = []
    for t in merged:
        ref = t["repeat_of"]
        target = None
        if ref:
            target = (id_map.get((t["chunk"], ref)) or id_map.get(("hist", ref))
                      or (ref if ref in hist_root else None))
            if target is None or (target in order and order[target] >= order[t["task_id"]]):
                stats["invalid_repeat_refs"] += 1
                warnings.append(f"{t['task_id']}: repeat_of {ref!r} is not an earlier task; cleared")
                target = None
        repeat_of = root.get(target, target) if target else None
        root[t["task_id"]] = repeat_of or t["task_id"]
        steps = [Step(order=i, description=s["description"], start=s["start"].to_pydatetime(),
                      end=s["end"].to_pydatetime(), evidence_segment_ids=s["evidence"])
                 for i, s in enumerate(t["steps"], 1)]
        out.append(PredictedTask(task_id=t["task_id"], label=t["label"], start=t["start"].to_pydatetime(),
                                 end=t["end"].to_pydatetime(), summary=t["summary"], steps=steps,
                                 repeat_of=repeat_of, confidence=t["confidence"],
                                 evidence_segment_ids=t["evidence"]))
    return out
