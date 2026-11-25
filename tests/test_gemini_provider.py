from types import SimpleNamespace

import pytest
from google.genai import errors

from workflow_recon.config import load_dotenv
from workflow_recon.llm import LLMConfigError, LLMError, LLMRefusal, get_provider
from workflow_recon.llm.gemini_provider import GeminiProvider


def _resp(text='{"tasks": []}', finish="STOP", block=None):
    return SimpleNamespace(
        text=text, model_version="gemini-3.8-flash",
        prompt_feedback=SimpleNamespace(block_reason=block) if block else None,
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name=finish))],
        usage_metadata=SimpleNamespace(prompt_token_count=100, candidates_token_count=20,
                                       thoughts_token_count=30, cached_content_token_count=0))


def _client(result, seen=None):
    def generate_content(**kw):
        if seen is not None:
            seen.update(kw)
        if isinstance(result, Exception):
            raise result
        return result
    return SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))


def _err(code, msg="x"):
    return errors.ClientError(code, {"error": {"code": code, "message": msg, "status": "X"}})


def test_request_shape_and_usage():
    seen = {}
    p = GeminiProvider("gemini-3.8-flash", seed=0, client=_client(_resp(), seen))
    r = p.generate_json("SYS", "USER", {"type": "object"}, 1234)
    assert r.data == {"tasks": []}
    assert r.usage == {"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 0}
    assert seen["model"] == "gemini-3.8-flash" and seen["contents"] == "USER"
    cfg = seen["config"]
    assert cfg.system_instruction == "SYS"
    assert cfg.response_mime_type == "application/json"
    assert cfg.response_json_schema == {"type": "object"}
    assert cfg.max_output_tokens == 1234 and cfg.seed == 0
    assert cfg.temperature is None          # model default unless configured


@pytest.mark.parametrize("resp,exc", [
    (_resp(finish="SAFETY"), LLMRefusal),
    (_resp(block=SimpleNamespace(name="PROHIBITED_CONTENT")), LLMRefusal),
    (_resp(finish="MAX_TOKENS"), LLMError),
    (_resp(text="not json"), LLMError),
    (_resp(text=None), LLMError),
])
def test_bad_responses(resp, exc):
    with pytest.raises(exc):
        GeminiProvider("m", client=_client(resp)).generate_json("s", "u", {}, 10)


def test_error_mapping():
    with pytest.raises(LLMConfigError):
        GeminiProvider("m", client=_client(_err(400, "API key not valid. Please pass a valid API key."))).generate_json("s", "u", {}, 1)
    with pytest.raises(LLMConfigError):
        GeminiProvider("m", client=_client(_err(404))).generate_json("s", "u", {}, 1)
    e = None
    try:
        GeminiProvider("m", client=_client(_err(429))).generate_json("s", "u", {}, 1)
    except Exception as ex:  # noqa: BLE001
        e = ex
    assert isinstance(e, LLMError) and not isinstance(e, LLMConfigError)


def test_missing_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(LLMConfigError):
        GeminiProvider("m").generate_json("s", "u", {}, 1)


def test_get_provider_reads_subsection(cfg):
    assert cfg["llm"]["provider"] == "gemini"
    p = get_provider(cfg["llm"])
    assert isinstance(p, GeminiProvider) and p.model == cfg["llm"]["gemini"]["model"]
    cfg["llm"]["provider"] = "anthropic"
    assert get_provider(cfg["llm"]).model == cfg["llm"]["anthropic"]["model"]


def test_llm_method_records_provider(day1, cfg):
    from workflow_recon.methods.llm_inference import run_llm
    p = GeminiProvider("gemini-3.8-flash", client=_client(_resp()))
    run, _ = run_llm(day1.segments, day1.window, cfg, p)
    assert run.params["llm"]["provider"] == "gemini" and run.params["llm"]["max_tokens"] == 65536
    assert run.params["usage"]["input_tokens"] == 100


def test_dotenv(tmp_path, monkeypatch):
    monkeypatch.delenv("WR_TEST_KEY", raising=False)
    monkeypatch.setenv("WR_TEST_SHELL", "from-shell")
    f = tmp_path / ".env"
    f.write_text('# comment\nexport WR_TEST_KEY="abc"\nWR_TEST_SHELL=from-file\nEMPTY=\n')
    loaded = load_dotenv(f)
    import os
    assert loaded == ["WR_TEST_KEY"] and os.environ["WR_TEST_KEY"] == "abc"
    assert os.environ["WR_TEST_SHELL"] == "from-shell"   # shell wins
    monkeypatch.delenv("WR_TEST_KEY")
