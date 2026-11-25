"""End-to-end orchestration: events -> window -> clean -> redact -> methods -> saved runs.

The task log is used here for exactly one thing: dropping time inside tasks
labelled "Private" (PROJECT_CONTEXT.md §5). Nothing else from it reaches a method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .clean import CleanResult, clean_events
from .config import resolve_path
from .io import load_history, save_run
from .llm import LLMProvider, get_provider
from .llm.cache import CachedProvider
from .methods.baseline import run_baseline
from .methods.llm_inference import run_llm
from .methods.sequence_mining import run_sequence_mining
from .redact import RedactionResult, redact_segments
from .schema import MethodRun
from .windows import Window, slice_events

ALL_METHODS = ("sequence_mining", "llm", "baseline")


@dataclass
class PreparedWindow:
    window: Window
    clean: CleanResult
    redaction: RedactionResult

    @property
    def segments(self) -> pd.DataFrame:
        """Redacted segments: what every method receives."""
        return self.redaction.segments


def private_intervals(task_log: pd.DataFrame | None) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if task_log is None or task_log.empty:
        return []
    priv = task_log[task_log["label"].str.strip().str.lower() == "private"]
    return list(zip(priv["start_utc"], priv["end_utc"]))


def prepare_window(events: pd.DataFrame, window: Window, cfg: dict,
                   private: Iterable[tuple[pd.Timestamp, pd.Timestamp]] = ()) -> PreparedWindow:
    sliced = slice_events(events, window)
    cleaned = clean_events(sliced, cfg, private)
    terms = resolve_path(cfg, "redact_terms_file") if cfg["paths"].get("redact_terms_file") else None
    red = redact_segments(cleaned.segments, cfg, terms)
    return PreparedWindow(window, cleaned, red)


def make_provider(cfg: dict) -> LLMProvider:
    lc = cfg["llm"]
    prov = get_provider(lc)
    if lc.get("cache", True):
        prov = CachedProvider(prov, resolve_path(cfg, "cache_dir") / "llm", lc["prompt_version"])
    return prov


def run_methods(prepared: PreparedWindow, cfg: dict, methods: Iterable[str] = ALL_METHODS,
                provider: LLMProvider | None = None, save: bool = True,
                runs_dir=None) -> dict[str, MethodRun]:
    runs_dir = runs_dir or resolve_path(cfg, "runs_dir")
    w = prepared.window
    segs = prepared.segments
    use_hist = cfg["sequence_mining"].get("use_history", True)
    results: dict[str, MethodRun] = {}
    for m in methods:
        sidecar = None
        if m == "sequence_mining":
            hist = load_history(runs_dir, m, w.start.to_pydatetime()) if use_hist else []
            run, sidecar = run_sequence_mining(segs, w, cfg, hist)
        elif m == "llm":
            hist = load_history(runs_dir, m, w.start.to_pydatetime())
            run, sidecar = run_llm(segs, w, cfg, provider or make_provider(cfg), hist)
        elif m == "baseline":
            run = run_baseline(segs, w, cfg)
        else:
            raise ValueError(f"unknown method {m!r}")
        run.params["preprocess"] = {"clean_stats": prepared.clean.stats,
                                    "redaction": prepared.redaction.report}
        if save:
            save_run(run, runs_dir, sidecar)
        results[m] = run
    return results
