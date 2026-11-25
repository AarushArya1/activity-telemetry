import pandas as pd

from workflow_recon.load import domain_of, inspect_export, load_events, load_task_log, source_for_bucket


def test_bucket_prefix_matching():
    assert source_for_bucket("aw-watcher-window_wifi-10-45-132-90.wifi.berkeley.edu") == "window"
    assert source_for_bucket("aw-watcher-web-chrome_other-host") == "web"
    assert source_for_bucket("aw-watcher-afk_x") == "afk"
    assert source_for_bucket("aw-watcher-jetbrains_x") == "editor"
    assert source_for_bucket("something-else", "app.editor.activity") == "editor"
    assert source_for_bucket("aw-stopwatch_x") is None


def test_flatten_concatenates_hosts_and_sorts(events, raw_export):
    hosts = {b["hostname"] for b in raw_export["buckets"].values()}
    assert len(hosts) == 2
    assert set(events["bucket_id"].str.split("_").str[1]) == hosts
    # stored newest-first, flattened oldest-first
    assert events["start_utc"].is_monotonic_increasing
    assert events["event_id"].is_unique
    assert set(events["source"]) == {"window", "web", "afk", "editor"}
    assert (events["end_utc"] >= events["start_utc"]).all()
    assert str(events["start_utc"].dt.tz) == "UTC"
    assert str(events["start_local"].dt.tz) == "America/Los_Angeles"


def test_editor_fields(events):
    ed = events[events["source"] == "editor"]
    assert ed["project"].notna().all() and ed["language"].notna().all()


def test_inspect_reports_counts(raw_export):
    rows = inspect_export(raw_export)
    assert sum(r["n_events"] for r in rows) == sum(len(b["events"]) for b in raw_export["buckets"].values())


def test_domain_of():
    assert domain_of("https://www.LinkedIn.com/jobs/?x=1") == "linkedin.com"
    assert domain_of(None) is None


def test_parquet_cache_roundtrip(tmp_path, events):
    src = tmp_path / "export.json"
    src.write_text((__import__("pathlib").Path(__file__).parent / "fixtures" / "fake_export.json").read_text())
    a = load_events(src, cache_dir=tmp_path / "c")
    b = load_events(src, cache_dir=tmp_path / "c")  # served from parquet
    assert list(tmp_path.joinpath("c").glob("*.parquet"))
    pd.testing.assert_frame_equal(a[["event_id", "start_utc", "app"]], b[["event_id", "start_utc", "app"]])


def test_task_log_timezones(tmp_path):
    p = tmp_path / "log.csv"
    p.write_text("Task ID,Date,Label,Start,End\nT03,Wed 11/18,Internship application,22:55,23:30\n")
    log = load_task_log(p, default_year=2026)
    r = log.iloc[0]
    # brief's example: 22:55-23:30 Nov 18 Pacific == 06:55-07:30 Nov 19 UTC
    assert r.start_utc == pd.Timestamp("2026-11-19T06:55:00Z")
    assert r.end_utc == pd.Timestamp("2026-11-19T07:30:00Z")


def test_fake_task_log_loads(fake_log):
    assert len(fake_log) == 10
    assert fake_log["start_local"].iloc[0].tzinfo is not None
