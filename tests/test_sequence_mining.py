import pandas as pd

from workflow_recon.methods.common import prefixspan, seq_similarity, weighted_jaccard
from workflow_recon.methods.sequence_mining import run_sequence_mining
from workflow_recon.pipeline import run_methods

from .conftest import best_match


def _gt(fake_log, day):
    return fake_log[fake_log["date"] == day]


def test_recovers_fake_tasks_day1(day1, cfg, fake_log):
    run, _ = run_sequence_mining(day1.segments, day1.window, cfg)
    for r in _gt(fake_log, "2026-12-01").itertuples():
        ov, t = best_match(run.tasks, r.start_local, r.end_local)
        dur = (r.end_local - r.start_local).total_seconds()
        assert t is not None and ov >= 0.6 * dur, f"{r.task_id} not covered"
        assert t.label == r.label, f"{r.task_id}: {t.label} != {r.label}"


def test_break_splits_and_continuation_links(day1, cfg, fake_log):
    run, _ = run_sequence_mining(day1.segments, day1.window, cfg)
    gt = _gt(fake_log, "2026-12-01").set_index("task_id")
    _, before = best_match(run.tasks, gt.loc["F04", "start_local"], gt.loc["F04", "end_local"])
    _, brk = best_match(run.tasks, gt.loc["F05", "start_local"], gt.loc["F05", "end_local"])
    _, after = best_match(run.tasks, gt.loc["F06", "start_local"], gt.loc["F06", "end_local"])
    assert len({before.task_id, brk.task_id, after.task_id}) == 3
    assert brk.label == "Entertainment"
    assert after.repeat_of == before.task_id


def test_repeats_across_days_use_own_history(day1, day2, cfg, fake_log, tmp_path):
    runs_dir = tmp_path / "runs"
    run_methods(day1, cfg, ["sequence_mining"], runs_dir=runs_dir)
    res = run_methods(day2, cfg, ["sequence_mining"], runs_dir=runs_dir)["sequence_mining"]
    d1 = run_methods(day1, cfg, ["sequence_mining"], save=False, runs_dir=runs_dir)["sequence_mining"]
    gt = fake_log.set_index("task_id")
    for rep, orig in (("F07", "F02"), ("F09", "F01")):
        _, t2 = best_match(res.tasks, gt.loc[rep, "start_local"], gt.loc[rep, "end_local"])
        _, t1 = best_match(d1.tasks, gt.loc[orig, "start_local"], gt.loc[orig, "end_local"])
        assert t2.repeat_of == t1.task_id


def test_steps_grounded_and_bounded(day1, day2, cfg):
    for prep in (day1, day2):
        run, _ = run_sequence_mining(prep.segments, prep.window, cfg)
        ids = set(prep.segments["segment_id"])
        for t in run.tasks:
            assert 1 <= len(t.steps) <= cfg["sequence_mining"]["max_steps"]
            assert set(t.evidence_segment_ids) <= ids
            for s in t.steps:
                assert s.evidence_segment_ids and set(s.evidence_segment_ids) <= set(t.evidence_segment_ids)
                assert t.start <= s.start <= s.end <= t.end
            assert [s.order for s in t.steps] == list(range(1, len(t.steps) + 1))


def test_deterministic(day1, cfg):
    a, _ = run_sequence_mining(day1.segments, day1.window, cfg)
    b, _ = run_sequence_mining(day1.segments, day1.window, cfg)
    strip = lambda r: [t.model_dump() for t in r.tasks]  # noqa: E731
    assert strip(a) == strip(b)


def test_sidecar_has_dfg_and_patterns(day1, cfg):
    _, sc = run_sequence_mining(day1.segments, day1.window, cfg)
    assert sc["tasks"] and sc["dfg"]
    assert all(len(p["pattern"]) >= 2 for p in sc["patterns"])


def test_empty_window(cfg, day1):
    run, _ = run_sequence_mining(day1.segments.iloc[0:0], day1.window, cfg)
    assert run.tasks == [] and run.warnings


def test_prefixspan():
    seqs = [["a", "b", "c"], ["a", "c"], ["b", "c"], ["a", "b"]]
    pats = {tuple(p): s for p, s in prefixspan(seqs, 2)}
    assert pats[("a",)] == 3 and pats[("a", "c")] == 2 and pats[("b", "c")] == 2
    assert ("c", "a") not in pats


def test_similarity_helpers():
    assert weighted_jaccard({"a": 1}, {"a": 1}) == 1.0
    assert weighted_jaccard({"a": 1}, {"b": 1}) == 0.0
    A, B = frozenset({"x"}), frozenset({"y"})
    assert seq_similarity([A, B], [A, B]) == 1.0
    assert seq_similarity([A], [B]) == 0.0
