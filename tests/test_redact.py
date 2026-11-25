from workflow_recon.methods.llm_inference import render_segments
from workflow_recon.redact import Redactor, find_leaks


def _r(cfg, terms=()):
    return Redactor(cfg, terms)


def test_text_entities(cfg):
    r = _r(cfg)
    out = r.redact_text("Re: lunch? call (510) 555-0142 - alex.example@berkeley.edu - Order 9876543210")
    assert "<PHONE>" in out and "<EMAIL>" in out and "<ID>" in out
    assert "555" not in out and "@" not in out
    assert r.counts["EMAIL"] == 1 and r.counts["PHONE"] == 1


def test_url_ids_and_profiles(cfg):
    r = _r(cfg)
    assert r.redact_url("doordash.com/orders/9f8e7d6c5b4a39281706") == "doordash.com/orders/<ID>"
    assert r.redact_url("linkedin.com/in/jane-doe-123") == "linkedin.com/in/<PERSON>"
    assert r.redact_url("instagram.com/samexample") == "instagram.com/<PERSON>"
    assert r.redact_url("instagram.com/direct/t/17845123456789012") == "instagram.com/direct/t/<ID>"
    assert r.redact_url("cs61a.org/hw/hw09") == "cs61a.org/hw/hw09"  # short tokens kept
    assert r.redact_url("linkedin.com/jobs/search?keywords=intern") == "linkedin.com/jobs/search?keywords=intern"


def test_user_terms(cfg):
    r = _r(cfg, ["Jordan Rivera"])
    assert r.redact_text("Chat with jordan rivera today") == "Chat with <PERSON> today"


def test_segment_table_redaction(day1):
    segs = day1.segments
    assert segs.attrs["redacted"] is True
    msgs = segs[segs["app"] == "Messages"]
    assert (msgs["title_norm"] == "<CHAT>").all()
    dm = segs[segs["url_norm"].fillna("").str.startswith("instagram.com/direct")]
    assert (dm["title_norm"] == "<CHAT>").all()
    rep = day1.redaction.report["entities"]
    assert rep["EMAIL"] >= 1 and rep["CHAT"] >= 2 and rep["PERSON"] >= 1
    # tokens are recomputed after redaction (no handle leaks into the token)
    assert not segs["token"].str.contains("samexample").any()


def test_no_leaks_in_llm_input(day1, day2):
    for prep in (day1, day2):
        text = render_segments(prep.segments)
        assert find_leaks(text) == {}
        for needle in ("alex.example", "555-0142", "Jordan Rivera", "samexample", "Sam Example"):
            assert needle not in text


def test_find_leaks():
    assert find_leaks("mail a@b.co") == {"EMAIL": 1}
    assert find_leaks("nothing here") == {}


def test_area_code_survives_title_normalization(cfg):
    """Regression: unread-count stripping must not eat the area code of a phone number."""
    from workflow_recon.clean import normalize_title
    r = _r(cfg)
    out = r.redact_text(normalize_title("Re: call me at (510) 555-0142 - Mail"))
    assert "555" not in out and "<PHONE>" in out
    assert r.redact_text("reach me 555-0142 tonight") == "reach me <PHONE> tonight"
