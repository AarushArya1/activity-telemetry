"""Cross-day pattern mining (one event log per label, variants, DFG) on synthetic data."""

import builtins
import json

import pytest

from workflow_recon import patterns
from workflow_recon.patterns import mine_patterns, to_dot
from workflow_recon.pipeline import run_methods


@pytest.fixture
def saved(cfg, day1, day2, tmp_path):
    runs = tmp_path / "runs"
    for d in (day1, day2):  # oldest first, like --all-days
        run_methods(d, cfg, ["sequence_mining", "baseline"], save=True, runs_dir=runs)
    return runs


def test_traces_saved_in_sidecar(saved):
    sc = json.loads((saved / "sequence_mining" / "2026-12-01.sidecar.json").read_text())
    run = json.loads((saved / "sequence_mining" / "2026-12-01.json").read_text())
    assert set(sc["traces"]) == {t["task_id"] for t in run["tasks"]}
    tr = next(iter(sc["traces"].values()))
    assert tr["steps"] and tr["sites"]
    acts = [a for a, _ in tr["sites"]]
    assert all(a != b for a, b in zip(acts, acts[1:]))  # consecutive repeats collapsed


def test_one_log_per_label_and_variants(saved):
    res = mine_patterns(saved, "sequence_mining", "steps")
    assert res["runs"] == 2 and res["tasks"] == sum(r["cases"] for r in res["labels"].values())
    intern = res["labels"]["Internship application"]
    # F02 and its repeat F07 follow the same step order -> one variant covering both tasks
    assert intern["cases"] == 2 and intern["n_variants"] == 1 and intern["top_variant_share"] == 1.0
    assert intern["variants"][0]["days"] == ["2026-12-01", "2026-12-02"]
    assert intern["start_activities"] == {"app.joinhandshake.com": 2}
    for r in res["labels"].values():
        assert sum(v["count"] for v in r["variants"]) == r["cases"]
        assert sum(r["start_activities"].values()) == r["cases"]


def test_site_level_has_at_least_as_many_variants(saved):
    steps = mine_patterns(saved, "sequence_mining", "steps")["labels"]
    sites = mine_patterns(saved, "sequence_mining", "sites")["labels"]
    for label in steps:
        assert sites[label]["n_variants"] >= steps[label]["n_variants"]


def test_builtin_fallback_matches_pm4py(saved, monkeypatch):
    pytest.importorskip("pm4py")
    with_pm4py = mine_patterns(saved, "sequence_mining", "steps")
    real_import = builtins.__import__

    def no_pm4py(name, *a, **k):
        if name == "pm4py":
            raise ImportError
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_pm4py)
    builtin = mine_patterns(saved, "sequence_mining", "steps")
    for label, r in with_pm4py["labels"].items():
        b = builtin["labels"][label]
        assert r["engine"] == "pm4py" and b["engine"] == "builtin"
        for k in ("cases", "n_variants", "variants", "dfg", "start_activities", "end_activities"):
            assert r[k] == b[k], (label, k)


def test_old_runs_without_traces_are_skipped(saved):
    sc = saved / "sequence_mining" / "2026-12-01.sidecar.json"
    data = json.loads(sc.read_text())
    del data["traces"]
    sc.write_text(json.dumps(data))
    res = mine_patterns(saved, "sequence_mining", "steps")
    assert res["runs"] == 1 and res["runs_without_traces"] == ["2026-12-01"]


def test_dot_output(saved):
    r = mine_patterns(saved, "sequence_mining", "steps")["labels"]["Internship application"]
    dot = to_dot(r)
    assert dot.startswith("digraph") and '"app.joinhandshake.com" -> "linkedin.com" [label=2' in dot


def test_never_reads_task_log():
    src = open(patterns.__file__).read()
    assert "task_log" not in src.replace("The task log (ground\ntruth) is never read here.", "")
