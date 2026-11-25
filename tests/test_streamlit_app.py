"""Headless UI tests using Streamlit's AppTest (synthetic data only)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
FIX = Path(__file__).parent / "fixtures" / "fake_export.json"


def _app():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def test_loads_with_sample_and_lists_days():
    at = _app()
    assert at.sidebar.selectbox[0].options == ["2026-12-01", "2026-12-02"]
    assert at.sidebar.selectbox[0].value == "2026-12-02"
    assert any("Press **Run**" in i.value or "press **Run**" in i.value for i in at.info)


def test_sequence_mining_run_renders_tasks():
    at = _app()
    at.sidebar.selectbox[0].select("2026-12-01").run()
    at.sidebar.radio[0].set_value("Sequence mining").run()
    at.sidebar.button[0].click().run()
    assert not at.exception, at.exception
    assert any("task(s) reconstructed" in s.value for s in at.subheader)
    heads = [e.label for e in at.expander]
    assert any("Internship application" in h for h in heads)
    assert any("Entertainment" in h for h in heads)
    assert any("repeat of" in h for h in heads)


def test_llm_without_key_shows_message(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr("workflow_recon.config.load_dotenv", lambda *a, **k: [])
    at = _app()
    at.sidebar.radio[0].set_value("LLM").run()
    assert any("GEMINI_API_KEY" in w.value for w in at.sidebar.warning)
    at.sidebar.button[0].click().run()
    assert not at.exception, at.exception
    assert any("no Gemini API key" in e.value for e in at.error)


def test_popup_buttons_open():
    at = _app()
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert any("What the methods can see" in m.value for m in at.markdown)
    at = _app()
    at.button[1].click().run()
    assert any("Method A: sequence / process mining" in m.value for m in at.markdown)


def test_upload_flow():
    at = _app()
    at.sidebar.file_uploader[0].upload("aw-buckets-export.json", FIX.read_bytes(), "application/json").run()
    at.sidebar.checkbox[0].uncheck().run()
    assert not at.exception, at.exception
    at.sidebar.selectbox[0].select("2026-12-02").run()
    at.sidebar.button[0].click().run()
    assert not at.exception, at.exception
    assert any("CS 61B Homework/Projects" in e.label for e in at.expander)


def test_bad_upload_shows_error():
    at = _app()
    at.sidebar.file_uploader[0].upload("broken.json", b"{not json", "application/json").run()
    at.sidebar.checkbox[0].uncheck().run()
    assert not at.exception, at.exception
    assert any("Could not read this file" in e.value for e in at.sidebar.error)
