from workflow_recon.io import list_runs, load_history, load_run, save_run
from workflow_recon.methods.baseline import run_baseline
from workflow_recon.pipeline import run_methods
from workflow_recon.schema import MethodRun


def test_baseline_runs(day1, cfg):
    run = run_baseline(day1.segments, day1.window, cfg)
    assert run.method == "baseline" and run.tasks
    assert all(t.steps == [] for t in run.tasks)


def test_save_load_roundtrip(day1, cfg, tmp_path):
    res = run_methods(day1, cfg, ["sequence_mining", "baseline"], runs_dir=tmp_path)
    for m, run in res.items():
        paths = list_runs(tmp_path, m)
        assert [p.name for p in paths] == ["2026-12-01.json"]
        back = load_run(paths[0])
        assert isinstance(back, MethodRun)
        assert back.model_dump() == MethodRun.model_validate_json(run.model_dump_json()).model_dump()
        assert back.params["preprocess"]["redaction"]["entities"]
    assert (tmp_path / "sequence_mining" / "2026-12-01.sidecar.json").exists()


def test_history_only_from_earlier_windows(day1, day2, cfg, tmp_path):
    run_methods(day2, cfg, ["sequence_mining"], runs_dir=tmp_path)
    assert load_history(tmp_path, "sequence_mining", day1.window.start.to_pydatetime()) == []
    assert load_history(tmp_path, "sequence_mining", day2.window.end.to_pydatetime())
