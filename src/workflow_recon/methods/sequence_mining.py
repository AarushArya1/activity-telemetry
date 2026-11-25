"""Method A — deterministic sequence / process mining (no LLM).

Pipeline for one window:
  1. Label hints per segment from transparent rules (labels.py).
  2. Blocks: split at idle gaps >= idle_split_s.
  3. Resolve labels inside each block: ambiguous segments (YouTube, Claude,
     VS Code, unknown sites) inherit the surrounding label; short label runs
     (< interruption_max_s) are absorbed as interruptions.
  4. Tasks = runs of one label; long runs are split further at Jaccard
     change-points of the token mix (binary segmentation).
  5. Steps: time bins with a similar token mix are grouped, rapid
     back-and-forth collapses into one step, short steps are merged, and the
     count is capped at max_steps.
  6. Process mining: a directly-follows graph per task (PM4Py if installed)
     and PrefixSpan over step sequences across tasks (incl. earlier runs).
  7. Repeat detection: step-sequence edit similarity + token-mix similarity
     against earlier tasks with the same label (this window and earlier runs).
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from typing import Any

import pandas as pd

from ..labels import LENT, LOTHER
from ..schema import MethodRun, PredictedTask, Step
from ..windows import Window
from .common import (SegArrays, add_label_hints, add_vec, normalized, prefixspan, runs,
                     seq_similarity, weighted_jaccard)

METHOD = "sequence_mining"
METHOD_VERSION = "a-v1"

# Weak hints differ in informativeness: "Other" (unknown / general tools) says
# nothing, YouTube-as-Entertainment is only half-reliable (taxonomy: YouTube can
# be lectures or entertainment).
_WEAK_WEIGHT = {LOTHER: 0.0, LENT: 0.5}

_VERBS = [
    (("youtube.com/watch", "youtube.com", "vimeo.com", "netflix.com", "twitch.tv"), "Watch videos on"),
    (("VS Code", "IntelliJ IDEA", "PyCharm", "Xcode", "Cursor"), "Code in"),
    (("Terminal", "iTerm2", "Warp"), "Run commands in"),
    (("mail.google.com", "Mail", "Microsoft Outlook"), "Read email in"),
    (("Messages", "WhatsApp", "Slack", "Discord", "instagram.com/direct", "linkedin.com/messaging"), "Message on"),
    (("google.com/search",), "Search on"),
    (("docs.google.com/document", "blank.page"), "Write in"),
    (("claude.ai", "chatgpt.com", "gemini.google.com"), "Ask"),
]


def _verb(token: str) -> str:
    for keys, verb in _VERBS:
        if token in keys or token.split("/")[0] in keys:
            return verb
    return "Use"


# ------------------------------------------------------------ label resolution

def site_affinity(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """For each ambiguous site: seconds it spends directly next to strong-labeled segments,
    per label. E.g. VS Code that keeps alternating with cs61a.org has 61A affinity."""
    aff: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    sites, labels, strong, dur = list(df["site"]), list(df["hint_label"]), list(df["hint_strong"]), list(df["duration_s"])
    for i in range(len(df)):
        if strong[i]:
            continue
        for j in (i - 1, i + 1):
            if 0 <= j < len(df) and strong[j]:
                aff[sites[i]][labels[j]] += dur[i]
    return aff


def _resolve_block(df: pd.DataFrame, cfg: dict, affinity: dict | None = None) -> list[str]:
    labels = list(df["hint_label"])
    strong = list(df["hint_strong"])
    dur = list(df["duration_s"])
    sites = list(df["site"])
    affinity = affinity or {}
    n = len(labels)
    resolved = list(labels)
    absorb_max = cfg["ambiguous_absorb_max_s"]

    # (a) ambiguous runs inherit context
    i = 0
    while i < n:
        if strong[i]:
            i += 1
            continue
        j = i
        while j < n and not strong[j]:
            j += 1
        prev_lab = labels[i - 1] if i > 0 else None
        next_lab = labels[j] if j < n else None
        weights: dict[str, float] = defaultdict(float)
        for k in range(i, j):
            weights[labels[k]] += dur[k] * _WEAK_WEIGHT.get(labels[k], 1.0)
        run_dur = sum(dur[i:j])
        own = max(weights, key=weights.get) if any(weights.values()) else \
            Counter({l: d for l, d in zip(labels[i:j], dur[i:j])}).most_common(1)[0][0]
        cands = [c for c in (prev_lab, next_lab) if c]
        if cands and run_dur <= absorb_max:
            if prev_lab == next_lab or len(cands) == 1:
                lab = cands[0]
            elif own in cands:
                lab = own
            else:  # between two different labels: follow the sites' usual company
                score = {c: sum(affinity.get(sites[k], {}).get(c, 0.0) for k in range(i, j)) for c in cands}
                lab = next_lab if score[next_lab] > score[prev_lab] else prev_lab
        elif cands:
            # long ambiguous run: continue the context only if it carries supporting hints
            share = {c: sum(d for l, d in zip(labels[i:j], dur[i:j]) if l == c) / max(run_dur, 1e-9) for c in cands}
            best = max(share, key=share.get)
            lab = best if share[best] >= 0.2 else own
        else:
            lab = own
        for k in range(i, j):
            resolved[k] = lab
        i = j

    # (b) absorb short runs (interruptions) into neighbours, shortest first
    while True:
        rs = runs(resolved)
        if len(rs) <= 1:
            break
        spans = [sum(dur[a:b]) for a, b in rs]
        k = min(range(len(rs)), key=lambda x: spans[x])
        if spans[k] >= cfg["interruption_max_s"]:
            break
        a, b = rs[k]
        prev_r = rs[k - 1] if k > 0 else None
        next_r = rs[k + 1] if k + 1 < len(rs) else None
        if prev_r and next_r:
            if resolved[prev_r[0]] == resolved[next_r[0]]:
                target = resolved[prev_r[0]]
            else:
                target = resolved[prev_r[0]] if spans[k - 1] >= spans[k + 1] else resolved[next_r[0]]
        else:
            target = resolved[(prev_r or next_r)[0]]
        for x in range(a, b):
            resolved[x] = target
    return resolved


# ------------------------------------------------------------ change points

def _changepoints(arr: SegArrays, ts: float, te: float, cfg: dict, pauses: list[float]) -> list[float]:
    """Binary segmentation of a same-label run.

    A cut is only allowed at a pause in activity (gap >= changepoint_min_gap_s), because
    consecutive *steps* of one workflow also have disjoint token sets; a pause plus a
    sharp change in the token mix is what separates two tasks with the same label.
    """
    W, min_task, thr = cfg["changepoint_window_s"], cfg["min_task_s"], cfg["changepoint_threshold"]
    if te - ts < 2 * min_task:
        return []
    best_t, best_sim = None, thr
    for t in pauses:
        if not ts + min_task <= t <= te - min_task:
            continue
        left = normalized(arr.token_vector(max(ts, t - W), t))
        right = normalized(arr.token_vector(t, min(te, t + W)))
        if left and right:
            sim = weighted_jaccard(left, right)
            if sim < best_sim:
                best_t, best_sim = t, sim
    if best_t is None:
        return []
    return (_changepoints(arr, ts, best_t, cfg, pauses) + [best_t]
            + _changepoints(arr, best_t, te, cfg, pauses))


# ------------------------------------------------------------ steps

def _extract_steps(arr: SegArrays, ts: float, te: float, cfg: dict) -> list[dict]:
    bin_s, ctx = cfg["step_bin_s"], cfg.get("step_context_s", 0)
    bins = []
    t = ts
    while t < te:
        b = min(te, t + bin_s)
        v = arr.token_vector(t, b)
        if sum(v.values()) >= 5:
            # compare using a context window so rapid back-and-forth reads as one mix
            bins.append((t, b, v, arr.token_vector(max(ts, t - ctx), min(te, b + ctx))))
        t = b
    steps: list[dict] = []
    for a, b, v, vctx in bins:
        if steps and weighted_jaccard(normalized(steps[-1]["vec"]), normalized(vctx)) >= cfg["step_similarity"]:
            steps[-1]["end"] = b
            steps[-1]["vec"] = add_vec(steps[-1]["vec"], v)
        else:
            steps.append({"start": a, "end": b, "vec": dict(v)})

    def merge(i: int, j: int) -> None:  # merge steps[j] into steps[i] (adjacent)
        lo, hi = min(i, j), max(i, j)
        steps[lo] = {"start": steps[lo]["start"], "end": steps[hi]["end"],
                     "vec": add_vec(steps[lo]["vec"], steps[hi]["vec"])}
        del steps[hi]

    def sim(i: int, j: int) -> float:
        return weighted_jaccard(normalized(steps[i]["vec"]), normalized(steps[j]["vec"]))

    # short steps -> most similar neighbour
    while len(steps) > 1:
        active = [sum(s["vec"].values()) for s in steps]
        k = min(range(len(steps)), key=lambda x: active[x])
        if active[k] >= cfg["min_step_s"]:
            break
        nbrs = [x for x in (k - 1, k + 1) if 0 <= x < len(steps)]
        merge(max(nbrs, key=lambda x: sim(k, x)), k)
    # identical dominant-token sets next to each other -> one step
    i = 0
    while i + 1 < len(steps):
        if set(_dominant(steps[i]["vec"], cfg)) == set(_dominant(steps[i + 1]["vec"], cfg)):
            merge(i, i + 1)
        else:
            i += 1
    # cap the count
    while len(steps) > cfg["max_steps"]:
        k = max(range(len(steps) - 1), key=lambda x: sim(x, x + 1))
        merge(k, k + 1)
    return steps


def _dominant(vec: dict[str, float], cfg: dict, cap: int = 3) -> tuple[str, ...]:
    share = normalized(vec)
    dom = sorted((t for t, x in share.items() if x >= cfg["step_token_min_share"]), key=lambda t: -share[t])[:cap]
    return tuple(dom) if dom else ((max(share, key=share.get),) if share else ())


def _describe(dom: tuple[str, ...], titles: dict[str, str]) -> str:
    def q(t: str) -> str:
        title = titles.get(t, "")
        return f' ("{title[:60]}")' if title else ""
    if not dom:
        return "Idle"
    if len(dom) == 1:
        return f"{_verb(dom[0])} {dom[0]}{q(dom[0])}"
    names = [f"{t}{q(t)}" for t in dom]
    return "Alternate between " + ", ".join(names[:-1]) + " and " + names[-1]


# ------------------------------------------------------------ process mining

def _dfg(tasks_tokens: dict[str, list[str]]) -> tuple[dict[str, list], str]:
    """Directly-follows graph per task (consecutive duplicates collapsed)."""
    collapsed = {tid: [t for i, t in enumerate(seq) if i == 0 or seq[i - 1] != t] for tid, seq in tasks_tokens.items()}
    try:
        import pm4py  # type: ignore

        out = {}
        for tid, seq in collapsed.items():
            if len(seq) < 2:
                out[tid] = []
                continue
            log = pd.DataFrame({"case:concept:name": tid, "concept:name": seq,
                                "time:timestamp": pd.date_range("2000-01-01", periods=len(seq), freq="s", tz="UTC")})
            dfg, _, _ = pm4py.discover_dfg(log)
            out[tid] = sorted(([a, b, int(c)] for (a, b), c in dfg.items()), key=lambda x: -x[2])
        return out, "pm4py"
    except ImportError:
        out = {}
        for tid, seq in collapsed.items():
            c = Counter(zip(seq, seq[1:]))
            out[tid] = sorted(([a, b, n] for (a, b), n in c.items()), key=lambda x: -x[2])
        return out, "builtin"


# ------------------------------------------------------------ main

def run_sequence_mining(segments: pd.DataFrame, window: Window, cfg: dict,
                        history: list[dict] | None = None) -> tuple[MethodRun, dict[str, Any]]:
    """Run Method A on one window. Returns (MethodRun, sidecar) — the sidecar holds
    mined patterns, DFGs and per-task signatures used for repeat detection later."""
    t0 = time.time()
    p = cfg["sequence_mining"]
    run = MethodRun(method=METHOD, method_version=METHOD_VERSION,
                    window_start=window.start.to_pydatetime(), window_end=window.end.to_pydatetime(),
                    params={"sequence_mining": p, "clean": cfg["clean"], "window": cfg["window"]})
    sidecar: dict[str, Any] = {"tasks": [], "dfg": {}, "patterns": []}
    if segments.empty:
        run.warnings.append("no segments in window")
        return run, sidecar

    df = add_label_hints(segments).sort_values("start").reset_index(drop=True)
    # Steps and repeat signatures work at site level (domain or app); the finer
    # path token (linkedin.com/jobs vs /messaging) is only needed for labeling.
    df["site"] = [d if isinstance(d, str) and d else (a or "unknown") for d, a in zip(df["domain"], df["app"])]
    starts = [t.timestamp() for t in df["start"]]
    ends = [t.timestamp() for t in df["end"]]

    # blocks at idle gaps
    block, run_end = [0], ends[0]
    for i in range(1, len(df)):
        block.append(block[-1] + int(starts[i] - run_end >= p["idle_split_s"]))
        run_end = max(run_end, ends[i])
    df["block"] = block
    df["label"] = ""
    affinity = site_affinity(df)
    for _, g in df.groupby("block", sort=True):
        df.loc[g.index, "label"] = _resolve_block(g, p, affinity)

    # tasks = label runs within blocks, refined by change points
    raw_tasks: list[tuple[str, list[int]]] = []
    keys = list(zip(df["block"], df["label"]))
    for a, b in runs(keys):
        idx = list(range(a, b))
        sub = df.iloc[idx]
        arr = SegArrays.from_df(sub, "site")
        ts, te = starts[a], max(ends[a:b])
        pauses = [starts[i] for i in idx[1:] if starts[i] - max(ends[a:i]) >= p["changepoint_min_gap_s"]]
        cuts = _changepoints(arr, ts, te, p, pauses)
        bounds = [ts] + cuts + [te]
        for lo, hi in zip(bounds, bounds[1:]):
            part = [i for i in idx if lo <= (starts[i] + ends[i]) / 2 < hi or (hi == te and starts[i] >= lo)]
            part = sorted(set(part))
            if part:
                raw_tasks.append((df.at[a, "label"], part))

    wid = window.window_id
    tasks: list[PredictedTask] = []
    sigs: list[dict] = []
    tokens_by_task: dict[str, list[str]] = {}
    dropped = 0
    for label, idx in raw_tasks:
        sub = df.iloc[idx]
        ts, te = starts[idx[0]], max(ends[i] for i in idx)
        if sum(sub["duration_s"]) < p["min_task_s"]:
            dropped += 1
            continue
        arr = SegArrays.from_df(sub, "site")
        tid = f"A-{wid}-{len(tasks) + 1:03d}"

        # top title per token (by duration)
        titles: dict[str, str] = {}
        tt = sub.groupby(["site", "title_norm"], dropna=False)["duration_s"].sum().reset_index()
        for tok, g in tt.groupby("site"):
            titles[tok] = g.sort_values("duration_s", ascending=False)["title_norm"].iloc[0] or ""

        steps = []
        step_sigs = []
        for st in _extract_steps(arr, ts, te, p):
            ov = arr.overlapping(st["start"], st["end"])
            ev_idx = [i for i, o in ov if o >= 1.0] or [i for i, _ in ov]
            if not ev_idx:
                continue
            s_start = max(st["start"], min(arr.s[i] for i in ev_idx))
            s_end = min(st["end"], max(arr.e[i] for i in ev_idx))
            dom = _dominant(st["vec"], p)
            step_sigs.append(sorted(dom))
            steps.append(Step(order=len(steps) + 1, description=_describe(dom, titles),
                              start=_ts(s_start, cfg), end=_ts(s_end, cfg),
                              evidence_segment_ids=[arr.ids[i] for i in ev_idx]))

        strong_agree = float(sub.loc[sub["hint_strong"] & (sub["hint_label"] == label), "duration_s"].sum())
        weak_agree = float(sub.loc[~sub["hint_strong"] & (sub["hint_label"] == label), "duration_s"].sum())
        total = float(sub["duration_s"].sum())
        conf = round(min(1.0, (strong_agree + 0.5 * weak_agree) / total), 3) if total else None

        tok_share = normalized(sub.groupby("site")["duration_s"].sum().to_dict())
        top = sorted(tok_share, key=lambda t: -tok_share[t])[:4]
        summary = (f"{label}: {round((te - ts) / 60)} min, {len(steps)} step(s), mostly "
                   + ", ".join(top) + ".")
        tasks.append(PredictedTask(task_id=tid, label=label, start=_ts(ts, cfg), end=_ts(te, cfg),
                                   summary=summary, steps=steps, confidence=conf,
                                   evidence_segment_ids=list(sub["segment_id"])))
        sigs.append({"task_id": tid, "label": label, "start": _ts(ts, cfg).isoformat(),
                     "end": _ts(te, cfg).isoformat(), "steps": step_sigs,
                     "tokens": {k: round(v, 4) for k, v in tok_share.items()}})
        tokens_by_task[tid] = list(sub["token"])  # DFG keeps the finer tokens

    _detect_repeats(tasks, sigs, history or [], p)

    dfg, engine = _dfg(tokens_by_task)
    all_seqs = [["+".join(s) for s in h["steps"]] for h in (history or [])] + \
               [["+".join(s) for s in sg["steps"]] for sg in sigs]
    patterns = [{"pattern": pat, "support": sup}
                for pat, sup in prefixspan(all_seqs, p["prefixspan_min_support"]) if len(pat) >= 2]
    patterns.sort(key=lambda x: (-x["support"], -len(x["pattern"])))

    run.tasks = tasks
    run.params["dfg_engine"] = engine
    run.params["stats"] = {"segments": len(df), "blocks": int(df["block"].nunique()),
                           "tasks": len(tasks), "short_tasks_dropped": dropped,
                           "runtime_s": round(time.time() - t0, 3)}
    sidecar.update(tasks=sigs, dfg=dfg, patterns=patterns[:50])
    return run, sidecar


def _ts(epoch: float, cfg: dict):
    return pd.Timestamp(round(epoch, 6), unit="s", tz="UTC").tz_convert(cfg["timezone"]).floor("us").to_pydatetime()


def _detect_repeats(tasks: list[PredictedTask], sigs: list[dict], history: list[dict], p: dict) -> None:
    """Set repeat_of to the first instance of a similar earlier task (same label)."""
    root: dict[str, str] = {h["task_id"]: h.get("repeat_of") or h["task_id"] for h in history}
    pool: list[dict] = list(history)
    for task, sg in zip(tasks, sigs):
        steps = [frozenset(s) for s in sg["steps"]]
        best, best_sim, how = None, p["repeat_threshold"], ""
        for cand in pool:
            if cand["label"] != task.label:
                continue
            tok_sim = weighted_jaccard(sg["tokens"], cand["tokens"])
            sim = 0.5 * seq_similarity(steps, [frozenset(s) for s in cand["steps"]]) + 0.5 * tok_sim
            gap = (pd.Timestamp(sg["start"]) - pd.Timestamp(cand["end"])).total_seconds()
            continuation = 0 <= gap <= p["continuation_gap_s"] and tok_sim >= 0.3
            if continuation and (best is None or how != "continuation"):
                best, best_sim, how = cand, max(sim, best_sim), "continuation"
            elif sim >= best_sim and how != "continuation":
                best, best_sim, how = cand, sim, "repeat"
        if best is not None:
            task.repeat_of = root.get(best["task_id"], best["task_id"])
            task.summary += (f" Continues {best['task_id']}." if how == "continuation"
                             else f" Similar to {best['task_id']} (sim {best_sim:.2f}).")
        root[task.task_id] = task.repeat_of or task.task_id
        sg["repeat_of"] = task.repeat_of
        pool.append(sg)
