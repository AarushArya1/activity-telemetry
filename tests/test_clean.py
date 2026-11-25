import pandas as pd

from workflow_recon.clean import clean_events, make_token, normalize_title, normalize_url
from workflow_recon.windows import day_window, slice_events

ALLOW = ["q", "keywords", "search_query"]


def test_normalize_url_strips_tracking_keeps_allowlist():
    d, p, u = normalize_url("https://www.LinkedIn.com/jobs/search/?currentJobId=4123&keywords=intern&trackingId=abc&refId=x#frag", ALLOW)
    assert d == "linkedin.com" and p == "/jobs/search"
    assert u == "linkedin.com/jobs/search?keywords=intern"


def test_normalize_url_hash_routes():
    _, p, _ = normalize_url("http://localhost:5600/#/timeline", ALLOW)
    assert p.endswith("#/timeline")


def test_normalize_title():
    assert normalize_title("Inbox (16,221) - someone - Mail") == "Inbox - someone - Mail"
    assert normalize_title("Joining Tables - YouTube", "youtube.com") == "Joining Tables"
    assert normalize_title("Jobs | LinkedIn", "linkedin.com") == "Jobs"
    assert normalize_title("Doc - Google Chrome") == "Doc"
    assert normalize_title("● hw09.py — hw09") == "hw09.py — hw09"
    assert normalize_title(None) == ""


def test_make_token():
    assert make_token("Google Chrome", "linkedin.com", "/jobs/view/123") == "linkedin.com/jobs"
    assert make_token("Google Chrome", "cs61a.org", "/") == "cs61a.org"
    assert make_token("VS Code", None, None) == "VS Code"


def test_afk_masking_removes_tab_left_open(day1, events, cfg):
    """The fixture keeps a YouTube tab logging during a 25-min AFK at ~09:39-10:04."""
    segs = day1.clean.segments
    afk = events[(events.source == "afk") & (events.status == "afk")]
    for r in afk.itertuples():
        inside = segs[(segs["start"] < r.end_utc) & (segs["end"] > r.start_utc)]
        assert inside.empty, f"segment overlaps AFK {r.start_local}"


def test_loginwindow_and_incognito_dropped(day1):
    segs = day1.clean.segments
    assert not (segs["app"] == "loginwindow").any()
    assert not segs["title_norm"].str.contains("secret").any()
    assert day1.clean.stats["incognito_dropped"] == 1


def test_chrome_dedup_uses_web_url(day1):
    chrome = day1.clean.segments[day1.clean.segments["app"] == "Google Chrome"]
    assert chrome["domain"].notna().mean() > 0.95


def test_micro_clicks_absorbed(day1):
    segs = day1.clean.segments
    assert (segs["duration_s"] >= 2).all()
    li = segs[segs["domain"] == "linkedin.com"]
    assert len(li) < 10  # 25 sub-second job-listing clicks collapse


def test_segments_sorted_and_have_sources(day1):
    segs = day1.clean.segments
    assert segs["start"].is_monotonic_increasing
    assert segs["segment_id"].is_unique
    assert segs["source_event_ids"].map(len).min() >= 1


def test_private_intervals_dropped(events, cfg):
    w = day_window("2026-12-01", cfg["timezone"])
    ps = pd.Timestamp("2026-12-01 10:05", tz=cfg["timezone"])
    pe = pd.Timestamp("2026-12-01 10:29", tz=cfg["timezone"])
    res = clean_events(slice_events(events, w), cfg, [(ps, pe)])
    segs = res.segments
    assert segs[(segs["start"] < pe) & (segs["end"] > ps)].empty


def test_editor_detail_attached(day2):
    ij = day2.clean.segments[day2.clean.segments["app"] == "IntelliJ IDEA"]
    assert ij["detail"].fillna("").str.contains("Java").any()
