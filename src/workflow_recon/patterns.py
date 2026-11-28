"""Cross-day process mining over a method's OWN saved runs: one event log per label.

Every predicted task becomes a *case*; its ordered activities form a *trace*.
For each label we build a PM4Py event log from all saved days and report:

  * variants  - the distinct traces and how many tasks followed each,
  * a directly-follows graph (DFG) with start / end activities.

Two activity levels:
  steps - one activity per step: the step's main site(s), e.g. "VS Code + cs61a.org"
  sites - every site switch inside the task (very fine; most traces end up unique)

Inputs are predicted tasks and redacted site tokens only. The task log (ground
truth) is never read here.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .io import list_runs, load_run
from .schema import MethodRun

LEVELS = ("steps", "sites")
STEP_SITE_MIN_SHARE = 0.25  # a site names a step if it holds >= 25% of the step's evidence time
STEP_SITE_MAX = 2


# ------------------------------------------------------------ traces (computed when a run is saved)

def _site_col(segments: pd.DataFrame) -> pd.Series:
    return pd.Series([d if isinstance(d, str) and d else (a if isinstance(a, str) and a else "unknown")
                      for d, a in zip(segments["domain"], segments["app"])], index=segments.index)


def _collapse(seq: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Drop consecutive repeats of the same activity (keeps the first timestamp)."""
    return [x for i, x in enumerate(seq) if i == 0 or seq[i - 1][0] != x[0]]


def step_activity(sites: pd.Series, durations: pd.Series) -> str:
    share = durations.groupby(sites.values).sum()
    share = share / share.sum() if share.sum() > 0 else share
    top = [s for s, v in share.sort_values(ascending=False).items() if v >= STEP_SITE_MIN_SHARE][:STEP_SITE_MAX]
    return " + ".join(sorted(top or [share.idxmax()]))


def task_traces(run: MethodRun, segments: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Per task: step-level and site-level traces as [activity, iso_start] pairs.

    The same definition is used for every method, so their patterns are comparable."""
    if segments.empty:
        return {}
    segs = segments.set_index("segment_id")
    site = _site_col(segments).set_axis(segments["segment_id"])
    out: dict[str, dict[str, Any]] = {}
    for t in run.tasks:
        steps = []
        for s in sorted(t.steps, key=lambda s: s.order):
            ids = [i for i in s.evidence_segment_ids if i in segs.index]
            if ids:
                steps.append((step_activity(site.loc[ids], segs.loc[ids, "duration_s"]), s.start.isoformat()))
        ids = [i for i in t.evidence_segment_ids if i in segs.index]
        sub = segs.loc[ids].assign(site=site.loc[ids].values).sort_values("start")
        sites = [(r.site, r.start.isoformat()) for r in sub.itertuples()]
        out[t.task_id] = {"label": t.label, "start": t.start.isoformat(), "end": t.end.isoformat(),
                          "steps": [list(x) for x in _collapse(steps)],
                          "sites": [list(x) for x in _collapse(sites)]}
    return out


# ------------------------------------------------------------ event log

def load_event_log(runs_dir: str | Path, method: str, level: str = "steps") -> tuple[pd.DataFrame, dict]:
    """All saved runs of `method` -> one PM4Py-style event log (+ info on what was skipped)."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    rows, info = [], {"runs": 0, "runs_without_traces": [], "tasks": 0}
    for p in list_runs(runs_dir, method):
        sc = p.with_suffix(".sidecar.json")
        traces = json.loads(sc.read_text()).get("traces") if sc.exists() else None
        if traces is None:
            info["runs_without_traces"].append(p.stem)
            continue
        info["runs"] += 1
        run = load_run(p)
        for t in run.tasks:
            tr = traces.get(t.task_id)
            if not tr or not tr[level]:
                continue
            info["tasks"] += 1
            prev = None
            for act, ts in tr[level]:
                ts = pd.Timestamp(ts)
                if prev is not None and ts <= prev:  # PM4Py orders by time; keep trace order strict
                    ts = prev + pd.Timedelta(microseconds=1)
                prev = ts
                rows.append({"case:concept:name": t.task_id, "concept:name": act, "time:timestamp": ts,
                             "label": t.label, "day": p.stem,
                             "case_duration_s": (t.end - t.start).total_seconds()})
    log = pd.DataFrame(rows, columns=["case:concept:name", "concept:name", "time:timestamp", "label", "day",
                                      "case_duration_s"])
    if not log.empty:
        log["time:timestamp"] = pd.to_datetime(log["time:timestamp"], utc=True)
    return log, info


# ------------------------------------------------------------ mining

def _discover(log: pd.DataFrame) -> tuple[dict, dict, dict, dict, str]:
    """(dfg, start_activities, end_activities, variant_counts, engine)."""
    try:
        import pm4py  # type: ignore

        dfg, starts, ends = pm4py.discover_dfg(log)
        variants = pm4py.get_variants(log)
        variants = {k: (v if isinstance(v, int) else len(v)) for k, v in variants.items()}
        return dict(dfg), dict(starts), dict(ends), variants, "pm4py"
    except ImportError:
        traces = log.groupby("case:concept:name", sort=False)["concept:name"].apply(tuple)
        dfg = Counter(p for tr in traces for p in zip(tr, tr[1:]))
        return (dict(dfg), dict(Counter(tr[0] for tr in traces)), dict(Counter(tr[-1] for tr in traces)),
                dict(Counter(traces)), "builtin")


def mine_label(log: pd.DataFrame) -> dict[str, Any]:
    """Variants + DFG for the cases of one label."""
    log = log.sort_values(["case:concept:name", "time:timestamp"])
    dfg, starts, ends, counts, engine = _discover(log)
    cases = log.groupby("case:concept:name", sort=False).agg(trace=("concept:name", tuple),
                                                            day=("day", "first"),
                                                            dur=("case_duration_s", "first"))
    n = len(cases)
    variants = []
    for trace, g in cases.groupby("trace", sort=False):
        variants.append({"activities": list(trace), "count": int(counts.get(trace, len(g))),
                         "share": round(len(g) / n, 3), "mean_duration_min": round(g["dur"].mean() / 60, 1),
                         "days": sorted(set(g["day"])), "task_ids": list(g.index)})
    variants.sort(key=lambda v: (-v["count"], v["activities"]))
    return {"cases": n, "n_variants": len(variants),
            "top_variant_share": variants[0]["share"] if variants else 0.0,
            "variants": variants,
            "dfg": sorted(([a, b, int(c)] for (a, b), c in dfg.items()), key=lambda e: (-e[2], e[0], e[1])),
            "start_activities": dict(sorted(starts.items(), key=lambda kv: -kv[1])),
            "end_activities": dict(sorted(ends.items(), key=lambda kv: -kv[1])),
            "engine": engine}


def mine_patterns(runs_dir: str | Path, method: str, level: str = "steps",
                  labels: list[str] | None = None) -> dict[str, Any]:
    """One event log per label across every saved day of `method`."""
    log, info = load_event_log(runs_dir, method, level)
    out: dict[str, Any] = {"method": method, "level": level, **info, "labels": {}}
    if log.empty:
        return out
    for label, g in log.groupby("label", sort=True):
        if labels and label not in labels:
            continue
        out["labels"][label] = mine_label(g)
    out["labels"] = dict(sorted(out["labels"].items(), key=lambda kv: -kv[1]["cases"]))
    return out


def save_patterns(result: dict[str, Any], out_dir: str | Path) -> Path:
    path = Path(out_dir) / f"{result['method']}_{result['level']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    return path


# ------------------------------------------------------------ rendering helpers

def variant_text(activities: list[str]) -> str:
    return " → ".join(activities)


def to_dot(label_result: dict[str, Any], max_edges: int = 40) -> str:
    """Graphviz DOT for a label's DFG (rendered by st.graphviz_chart, no binary needed)."""
    def q(s: str) -> str:
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'

    edges = label_result["dfg"][:max_edges]
    nodes = Counter()
    for a, b, c in edges:
        nodes[a] += c
        nodes[b] += c
    for a in label_result["start_activities"]:
        nodes[a] += 0
    for a in label_result["end_activities"]:
        nodes[a] += 0
    top = max([c for *_, c in edges] or [1])
    lines = ["digraph DFG {", "  rankdir=LR;", '  node [shape=box, style="rounded,filled", fillcolor="#eef3fa",'
             ' color="#4e79a7", fontname="Helvetica", fontsize=11];', '  edge [color="#666666", fontsize=10];',
             '  "▶ start" [shape=circle, label="▶", fillcolor="#59a14f", fontcolor=white, width=0.3];',
             '  "■ end" [shape=doublecircle, label="", fillcolor="#e15759", width=0.2];']
    for n in nodes:
        lines.append(f"  {q(n)};")
    for a, c in label_result["start_activities"].items():
        if a in nodes:
            lines.append(f'  "▶ start" -> {q(a)} [label={c}, style=dashed];')
    for a, c in label_result["end_activities"].items():
        if a in nodes:
            lines.append(f'  {q(a)} -> "■ end" [label={c}, style=dashed];')
    for a, b, c in edges:
        lines.append(f"  {q(a)} -> {q(b)} [label={c}, penwidth={1 + 3 * c / top:.1f}];")
    lines.append("}")
    return "\n".join(lines)


def summarize_text(result: dict[str, Any], top: int = 5) -> str:
    lines = [f"method={result['method']} level={result['level']} runs={result['runs']} tasks={result['tasks']}"]
    if result["runs_without_traces"]:
        lines.append(f"  skipped {len(result['runs_without_traces'])} run(s) saved before traces existed "
                     f"(re-run them): {', '.join(result['runs_without_traces'])}")
    for label, r in result["labels"].items():
        lines.append(f"\n## {label}: {r['cases']} task(s), {r['n_variants']} variant(s), "
                     f"top variant covers {r['top_variant_share']:.0%}  [{r['engine']}]")
        for v in r["variants"][:top]:
            lines.append(f"  {v['count']:>3}x {v['share']:>4.0%}  ~{v['mean_duration_min']} min  "
                         f"{variant_text(v['activities'])}")
        if r["n_variants"] > top:
            lines.append(f"  … {r['n_variants'] - top} more")
        if r["dfg"]:
            lines.append("  most common hand-offs: "
                         + "; ".join(f"{a} → {b} ({c})" for a, b, c in r["dfg"][:5]))
    return "\n".join(lines)
