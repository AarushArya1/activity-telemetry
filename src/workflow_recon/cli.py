"""Command line entry point.

  python -m workflow_recon inspect [--export PATH]
  python -m workflow_recon run [--export PATH] [--date YYYY-MM-DD | --all-days]
                               [--methods sequence_mining,llm,baseline] [--mode day|session]
  python -m workflow_recon patterns [--method sequence_mining] [--level steps|sites] [--label L]
"""

from __future__ import annotations

import argparse
import sys

from .config import load_config, resolve_path
from .llm import LLMConfigError
from .patterns import LEVELS, mine_patterns, save_patterns, summarize_text
from .load import inspect_export, load_events, load_task_log, read_export
from .pipeline import ALL_METHODS, make_provider, prepare_window, private_intervals, run_methods
from .windows import select_windows


def _cmd_inspect(args, cfg) -> int:
    path = args.export or resolve_path(cfg, "export")
    for row in inspect_export(read_export(path)):
        print(f"{row['bucket_id']:<60} source={row['source']!s:<7} n={row['n_events']:<7} "
              f"{row['first']} .. {row['last']}\n    keys={row['data_keys']}")
    return 0


def _cmd_run(args, cfg) -> int:
    if args.mode:
        cfg["window"]["mode"] = args.mode
    path = args.export or resolve_path(cfg, "export")
    events = load_events(path, cfg["timezone"], resolve_path(cfg, "cache_dir"))
    tl_path = resolve_path(cfg, "task_log")
    # The task log is read ONLY to drop "Private" time. It is never passed to a method.
    priv = private_intervals(load_task_log(tl_path, cfg["timezone"])) if tl_path.exists() else []

    windows = select_windows(events, cfg, date=args.date)
    if not args.all_days and not args.date and windows:
        windows = windows[-1:]  # default: most recent day in the export
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    provider = make_provider(cfg) if "llm" in methods else None

    for w in windows:
        prep = prepare_window(events, w, cfg, priv)
        print(f"\n=== window {w.window_id} ({w.mode}) — {prep.clean.stats.get('segments', 0)} segments, "
              f"redactions={prep.redaction.report['entities']}")
        if prep.segments.empty:
            print("  (no activity)")
            continue
        results = run_methods(prep, cfg, methods, provider=provider, save=not args.no_save)
        for m, run in results.items():
            print(f"  [{m}] {len(run.tasks)} tasks" + (f", warnings: {run.warnings}" if run.warnings else ""))
            for t in run.tasks:
                rep = f" (repeat of {t.repeat_of})" if t.repeat_of else ""
                print(f"    {t.task_id} {t.start:%H:%M}–{t.end:%H:%M} {t.label}{rep}")
                if args.verbose:
                    for s in t.steps:
                        print(f"        {s.order}. {s.start:%H:%M}–{s.end:%H:%M} {s.description}")
    return 0


def _cmd_patterns(args, cfg) -> int:
    res = mine_patterns(resolve_path(cfg, "runs_dir"), args.method, args.level,
                        labels=[args.label] if args.label else None)
    if not res["tasks"]:
        print(f"no saved {args.method} runs with traces yet: run `workflow_recon run --all-days "
              f"--methods {args.method}` first")
        return 1
    print(summarize_text(res, top=args.top))
    print(f"\nsaved {save_patterns(res, resolve_path(cfg, 'patterns_dir'))}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="workflow_recon")
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_i = sub.add_parser("inspect", help="summarize buckets in an export")
    p_i.add_argument("--export")
    p_r = sub.add_parser("run", help="run methods over day/session windows")
    p_r.add_argument("--export")
    p_r.add_argument("--date", help="local date YYYY-MM-DD (default: most recent day)")
    p_r.add_argument("--all-days", action="store_true", help="run every day in the export, oldest first")
    p_r.add_argument("--methods", default=",".join(ALL_METHODS))
    p_r.add_argument("--mode", choices=["day", "session"], help="override window.mode")
    p_r.add_argument("--no-save", action="store_true")
    p_r.add_argument("-v", "--verbose", action="store_true", help="print steps")
    p_p = sub.add_parser("patterns", help="variants + DFG per label across all saved runs")
    p_p.add_argument("--method", default="sequence_mining", choices=list(ALL_METHODS))
    p_p.add_argument("--level", default="steps", choices=list(LEVELS))
    p_p.add_argument("--label", help="only this label")
    p_p.add_argument("--top", type=int, default=5, help="variants to print per label")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    try:
        return {"inspect": _cmd_inspect, "run": _cmd_run, "patterns": _cmd_patterns}[args.cmd](args, cfg)
    except LLMConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
