"""Workflow reconstruction demo.

Run:  .venv/bin/streamlit run app/streamlit_app.py

Upload an ActivityWatch export, pick a day and a method, and see the reconstructed workflow.
All heavy lifting lives in the workflow_recon package; this file only calls it.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from explainers import BENCHMARK_PLAN, CORE_CONCEPTS, PROJECT_OVERVIEW  # noqa: E402
from workflow_recon.config import load_config, resolve_path  # noqa: E402
from workflow_recon.llm import LLMConfigError, provider_settings  # noqa: E402
from workflow_recon.load import flatten_export  # noqa: E402
from workflow_recon.methods.llm_inference import PrivacyError  # noqa: E402
from workflow_recon.patterns import mine_patterns, save_patterns, to_dot, variant_text  # noqa: E402
from workflow_recon.pipeline import make_provider, prepare_window, run_methods  # noqa: E402
from workflow_recon.windows import available_days, day_window  # noqa: E402

SAMPLE = ROOT / "tests" / "fixtures" / "fake_export.json"
METHODS = {"Sequence mining": "sequence_mining", "LLM": "llm"}

st.set_page_config(page_title="Workflow reconstruction", page_icon="🧭", layout="wide")


# ------------------------------------------------------------ cached pipeline steps

@st.cache_data(show_spinner="Parsing export…", max_entries=4)
def parse_export(file_id: str, _raw: bytes, tz: str) -> pd.DataFrame:
    return flatten_export(json.loads(_raw), tz)


@st.cache_data(show_spinner="Cleaning and redacting…", max_entries=8)
def prepare(file_id: str, day: str, _events: pd.DataFrame, _cfg: dict):
    return prepare_window(_events, day_window(day, _cfg["timezone"]), _cfg)


# ------------------------------------------------------------ popups

@st.dialog("How this project works", width="large")
def show_overview() -> None:
    st.markdown(PROJECT_OVERVIEW)


@st.dialog("Core concepts & code", width="large")
def show_concepts() -> None:
    st.markdown(CORE_CONCEPTS)


@st.dialog("Benchmark plan (draft)", width="large")
def show_benchmark() -> None:
    st.markdown(BENCHMARK_PLAN)


# ------------------------------------------------------------ rendering

def timeline(run, segments: pd.DataFrame) -> alt.Chart:
    rows = [{"lane": "Activity", "start": r.start, "end": r.end, "what": r.token, "label": "activity"}
            for r in segments.itertuples()]
    rows += [{"lane": "Tasks", "start": t.start, "end": t.end, "what": t.summary, "label": t.label}
             for t in run.tasks]
    df = pd.DataFrame(rows)
    if df.empty:
        return alt.Chart(pd.DataFrame({"x": []})).mark_bar()
    # Altair renders in the browser's timezone; send naive local times so the axis shows Pacific.
    for c in ("start", "end"):
        df[c] = pd.to_datetime(df[c], utc=True).dt.tz_convert(segments["start"].iloc[0].tz).dt.tz_localize(None)
    labels = sorted(df.loc[df["lane"] == "Tasks", "label"].unique())
    color = alt.Color("label:N", title="Label",
                      scale=alt.Scale(domain=["activity"] + labels,
                                      range=["#c8c8c8"] + ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2",
                                                           "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]))
    return (alt.Chart(df).mark_bar(cornerRadius=2)
            .encode(x=alt.X("start:T", title=None, axis=alt.Axis(format="%H:%M")), x2="end:T",
                    y=alt.Y("lane:N", title=None, sort=["Tasks", "Activity"]), color=color,
                    tooltip=[alt.Tooltip("label:N"), alt.Tooltip("what:N", title="detail"),
                             alt.Tooltip("start:T", format="%H:%M:%S"), alt.Tooltip("end:T", format="%H:%M:%S")])
            .properties(height=140))


def show_run(run, prep) -> None:
    segs = prep.segments
    st.subheader(f"{len(run.tasks)} task(s) reconstructed")
    c = st.columns(4)
    c[0].metric("Raw events", sum(prep.clean.stats.get("events_in", {}).values()))
    c[1].metric("Segments after cleaning", len(segs))
    c[2].metric("Redactions", sum(prep.redaction.report["entities"].values()))
    usage = run.params.get("usage")
    if usage:
        c[3].metric("LLM tokens (in / out)", f"{usage.get('input_tokens', 0):,} / {usage.get('output_tokens', 0):,}")
    else:
        c[3].metric("Runtime", f"{run.params.get('stats', {}).get('runtime_s', 0)} s")

    st.altair_chart(timeline(run, segs), use_container_width=True)
    for w in run.warnings:
        st.warning(w)

    by_id = segs.set_index("segment_id")
    for t in run.tasks:
        head = f"**{t.start:%H:%M}–{t.end:%H:%M}** · {t.label}"
        if t.repeat_of:
            head += f" · 🔁 repeat of `{t.repeat_of}`"
        with st.expander(head, expanded=len(run.tasks) <= 6):
            st.markdown(t.summary)
            meta = f"`{t.task_id}` · {round((t.end - t.start).total_seconds() / 60)} min"
            if t.confidence is not None:
                meta += f" · confidence {t.confidence:.2f}"
            st.caption(meta)
            for s in t.steps:
                st.markdown(f"{s.order}. {s.description}  \n"
                            f"<span style='color:gray;font-size:0.85em'>{s.start:%H:%M}–{s.end:%H:%M} · "
                            f"{len(s.evidence_segment_ids)} evidence segment(s)</span>", unsafe_allow_html=True)
            if not t.steps:
                st.caption("No steps.")
            if st.toggle("Show evidence segments", key=f"ev-{t.task_id}"):
                ids = [i for s in t.steps for i in s.evidence_segment_ids] or t.evidence_segment_ids
                ev = by_id.loc[[i for i in dict.fromkeys(ids) if i in by_id.index]]
                st.dataframe(ev[["start", "end", "token", "title_norm", "url_norm"]]
                             .assign(start=lambda d: d["start"].dt.strftime("%H:%M:%S"),
                                     end=lambda d: d["end"].dt.strftime("%H:%M:%S")),
                             use_container_width=True)

    with st.expander("Diagnostics"):
        st.write("**Cleaning**", prep.clean.stats)
        st.write("**Redaction** (counts only)", prep.redaction.report)
        st.write("**Method params**", {k: v for k, v in run.params.items() if k != "preprocess"})
    st.download_button("Download run JSON", run.model_dump_json(indent=2),
                       file_name=f"{run.method}_{prep.window.window_id}.json", mime="application/json")


LEVEL_HELP = {"steps": "One activity per step (its main sites). Best for comparing how tasks are done.",
              "sites": "Every switch between sites. Very detailed; almost every task becomes its own variant."}


def run_all_days(raw_id, days, events, cfg, method, method_label) -> None:
    """Run `method` on every day (oldest first, so repeat detection sees earlier days) and save."""
    provider = make_provider(cfg) if method == "llm" else None
    bar = st.progress(0.0, text=f"Running {method_label.lower()} on {len(days)} day(s)…")
    for i, d in enumerate(days):
        prep = prepare(raw_id, d, events, cfg)
        if not prep.segments.empty:
            run_methods(prep, cfg, [method], provider=provider, save=True)
        bar.progress((i + 1) / len(days), text=f"{d} done")
    bar.empty()


def show_patterns(cfg, method, method_label, file_id, days, events) -> None:
    st.markdown(
        "Every **saved** run of the chosen method is pooled into **one event log per label**: each task is "
        "a *case*, its steps are the *activities*. PM4Py then finds the **variants** (distinct step orders) "
        "and a **directly-follows graph**. Only the method's own predicted tasks are used, never the task log.")
    c1, c2 = st.columns([2, 3])
    level = c1.radio("Activity level", ["steps", "sites"], horizontal=True, help=" ".join(
        f"**{k}**: {v}" for k, v in LEVEL_HELP.items()))
    c2.write("")
    if c2.button(f"Run & save {method_label.lower()} on all {len(days)} day(s) in this file",
                 help="Needed once per export so the patterns cover every day. "
                      + ("Uses one or more LLM calls per day (cached afterwards)." if method == "llm" else "")):
        try:
            run_all_days(file_id, days, events, cfg, method, method_label)
        except LLMConfigError as e:
            st.error(str(e))
            return
        except PrivacyError as e:
            st.error(f"Blocked before sending to the LLM: {e}")
            return

    res = mine_patterns(resolve_path(cfg, "runs_dir"), method, level)
    if res["runs_without_traces"]:
        st.caption(f"{len(res['runs_without_traces'])} older run(s) were saved before traces existed and are "
                   "skipped; the button above re-runs them.")
    if not res["tasks"]:
        st.info(f"No saved {method_label.lower()} runs yet. Press the button above (or tick "
                "*Save run* in the sidebar) to build them.")
        return
    save_patterns(res, resolve_path(cfg, "patterns_dir"))
    m = st.columns(3)
    m[0].metric("Saved days", res["runs"])
    m[1].metric("Tasks (cases)", res["tasks"])
    m[2].metric("Labels", len(res["labels"]))

    for label, r in res["labels"].items():
        head = (f"**{label}** · {r['cases']} task(s) · {r['n_variants']} variant(s) · "
                f"most common way: {r['top_variant_share']:.0%}")
        with st.expander(head, expanded=False):
            st.caption(f"{r['cases']} tasks were done in {r['n_variants']} distinct way(s). "
                       f"Fewer variants = a more routine workflow. Engine: {r['engine']}.")
            table = pd.DataFrame([{"Tasks": v["count"], "Share": f"{v['share']:.0%}",
                                   "Avg length (min)": v["mean_duration_min"],
                                   "Variant (step order)": variant_text(v["activities"]),
                                   "Days": ", ".join(v["days"]),
                                   "Task IDs": ", ".join(v["task_ids"])} for v in r["variants"]])
            st.dataframe(table, hide_index=True, use_container_width=True)
            if r["dfg"]:
                st.markdown("**Directly-follows graph** (arrow labels = how many times one activity came "
                            "straight after the other, across all tasks with this label)")
                st.graphviz_chart(to_dot(r), use_container_width=True)
            else:
                st.caption("Single-activity tasks only, so there is no graph to draw.")


# ------------------------------------------------------------ page

def main() -> None:
    cfg = load_config()
    tz = cfg["timezone"]

    st.title("🧭 Workflow reconstruction from activity telemetry")
    st.caption("Upload a day of ActivityWatch data, pick a method, and see the workflow it reconstructs.")
    b1, b2, b3, _ = st.columns([1, 1, 1, 3])
    if b1.button("ℹ️ How this project works", use_container_width=True):
        show_overview()
    if b2.button("📚 Core concepts & code", use_container_width=True):
        show_concepts()
    if b3.button("📏 Benchmark plan", use_container_width=True):
        show_benchmark()

    with st.sidebar:
        st.header("1 · Data")
        up = st.file_uploader("ActivityWatch export (.json)", type=["json"],
                              help="Raw data stays on this machine. Only redacted text is ever sent to an LLM.")
        use_sample = st.checkbox("Use synthetic sample data", value=up is None,
                                 help="Generated by scripts/make_fake_data.py; contains no real data.")
        raw = up.getvalue() if (up is not None and not use_sample) else (SAMPLE.read_bytes() if use_sample else None)
        if raw is None:
            st.info("Upload an export or tick the sample box.")
            st.stop()
        file_id = hashlib.sha1(raw).hexdigest()
        try:
            events = parse_export(file_id, raw, tz)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            st.error(f"Could not read this file as an ActivityWatch export: {e}")
            st.stop()
        days = [d.strftime("%Y-%m-%d") for d in available_days(events, tz)]
        if not days:
            st.error("No window-watcher activity found in this file.")
            st.stop()
        day = st.selectbox("Day", days, index=len(days) - 1, help="Times are Pacific.")

        st.header("2 · Method")
        method_label = st.radio("Method", list(METHODS), label_visibility="collapsed",
                                captions=["Deterministic rules + process mining. Instant, free.",
                                          f"{cfg['llm']['provider'].title()} · "
                                          f"{provider_settings(cfg['llm'])['model']}"])
        method = METHODS[method_label]
        if method == "llm":
            key_var = "GEMINI_API_KEY" if cfg["llm"]["provider"] == "gemini" else "ANTHROPIC_API_KEY"
            if not (os.environ.get(key_var) or (key_var == "GEMINI_API_KEY" and os.environ.get("GOOGLE_API_KEY"))):
                st.warning(f"No `{key_var}` found. Add it to the `.env` file in the project folder, "
                           "then restart the app.")
        save = st.checkbox("Save run to outputs/runs", value=False,
                           help="Saved runs let later days detect repeats of earlier workflows.")
        go = st.button("▶ Run", type="primary", use_container_width=True)

    tab_day, tab_pat = st.tabs(["📅 This day", "🔀 Patterns across days"])
    with tab_pat:
        show_patterns(cfg, method, method_label, file_id, days, events)
    with tab_day:
        show_day(cfg, file_id, day, events, method, method_label, go, save)


def show_day(cfg, file_id, day, events, method, method_label, go, save) -> None:
    prep = prepare(file_id, day, events, cfg)
    if prep.segments.empty:
        st.info("No activity on this day after cleaning.")
        return

    key = (file_id, day, method)
    if go:
        with st.spinner(f"Running {method_label.lower()}…"):
            try:
                provider = make_provider(cfg) if method == "llm" else None
                run = run_methods(prep, cfg, [method], provider=provider, save=save)[method]
            except LLMConfigError as e:
                st.error(str(e))
                return
            except PrivacyError as e:
                st.error(f"Blocked before sending to the LLM: {e}")
                return
        st.session_state["result"] = (key, run)

    result = st.session_state.get("result")
    if result and result[0] == key:
        show_run(result[1], prep)
    else:
        st.info(f"**{day}** · {len(prep.segments)} activity segments ready. Choose a method and press **Run**.")

main()
