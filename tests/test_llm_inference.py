import json
import re
from types import SimpleNamespace

import pytest

from workflow_recon.labels import PREDICTABLE_LABELS
from workflow_recon.llm import LLMRefusal, LLMResponse
from workflow_recon.llm.anthropic_provider import FALLBACK_BETA, AnthropicProvider
from workflow_recon.llm.cache import CachedProvider
from workflow_recon.methods.llm_inference import (PrivacyError, load_prompt, output_json_schema,
                                                  render_segments, run_llm)


class ScriptedProvider:
    name = "scripted"
    model = "scripted-1"

    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    def settings(self):
        return {"provider": self.name, "model": self.model}

    def generate_json(self, system, user, schema, max_tokens):
        self.calls.append(user)
        return LLMResponse(data=self.fn(system, user), model=self.model, usage={"input_tokens": len(user) // 4,
                                                                                 "output_tokens": 100})


def ids_in(user):
    return re.findall(r"^\[(S\d{4})\]", user, flags=re.M)


def ids_between(segs, start, end):
    m = (segs["start"] >= start) & (segs["end"] <= end)
    return list(segs.loc[m, "segment_id"])


def test_validation_drops_hallucinations(day1, cfg, fake_log):
    segs = day1.segments
    gt = fake_log.set_index("task_id")
    f02 = ids_between(segs, gt.loc["F02", "start_local"], gt.loc["F02", "end_local"])
    f04 = ids_between(segs, gt.loc["F04", "start_local"], gt.loc["F04", "end_local"])
    f06 = ids_between(segs, gt.loc["F06", "start_local"], gt.loc["F06", "end_local"])

    def fn(system, user):
        return {"tasks": [
            {"local_id": "t1", "label": "Internship application", "start_segment_id": f02[0],
             "end_segment_id": f02[-1], "summary": "Applied to internships.", "confidence": 0.9, "repeat_of": None,
             "steps": [{"description": "Search Handshake", "evidence_segment_ids": [f02[0], f02[1]]},
                       {"description": "Invented step", "evidence_segment_ids": ["S9999"]},
                       {"description": "Out of range", "evidence_segment_ids": [f04[0], f02[2]]}]},
            {"local_id": "t2", "label": "CS 61A Notes/Homework", "start_segment_id": f04[0],
             "end_segment_id": f04[-1], "summary": "Homework.", "confidence": 0.8, "repeat_of": None,
             "steps": [{"description": "Work in VS Code", "evidence_segment_ids": f04[:3]}]},
            {"local_id": "t3", "label": "CS 61A Notes/Homework", "start_segment_id": f06[0],
             "end_segment_id": f06[-1], "summary": "Continue homework.", "confidence": 0.8, "repeat_of": "t2",
             "steps": [{"description": "Finish and submit", "evidence_segment_ids": f06[-2:]}]},
            {"local_id": "t4", "label": "Other", "start_segment_id": "S8888", "end_segment_id": "S8889",
             "summary": "Ghost.", "confidence": 0.1, "repeat_of": "t9",
             "steps": [{"description": "Nothing", "evidence_segment_ids": ["S7777"]}]},
        ]}

    run, sidecar = run_llm(segs, day1.window, cfg, ScriptedProvider(fn))
    st = run.params["stats"]
    assert st["hallucinated_citations"] == 2       # S9999, S7777
    assert st["out_of_range_citations"] == 1       # f04[0] cited inside the F02 task
    assert st["hallucinated_steps"] == 2           # the invented step + t4's step (t4 dropped entirely)
    assert st["steps_total"] == 6
    assert st["tasks_dropped"] == 1
    assert [t.label for t in run.tasks] == ["Internship application", "CS 61A Notes/Homework", "CS 61A Notes/Homework"]
    t1, t2, t3 = run.tasks
    assert [s.description for s in t1.steps] == ["Search Handshake", "Out of range"]
    assert t1.steps[1].evidence_segment_ids == [f02[2]]
    assert t3.repeat_of == t2.task_id                # local id resolved to final id
    assert t2.task_id.startswith("B-2026-12-01-")
    all_ids = set(segs["segment_id"])
    for t in run.tasks:
        for s in t.steps:
            assert set(s.evidence_segment_ids) <= all_ids
            assert t.start <= s.start <= s.end <= t.end
    assert any("dropped" in w for w in run.warnings)
    assert run.params["usage"]["input_tokens"] > 0


def test_chunking_and_boundary_merge(day1, cfg):
    cfg["llm"]["max_segments_per_chunk"] = 30
    cfg["llm"]["chunk_overlap_segments"] = 5

    def fn(system, user):
        ids = ids_in(user)
        return {"tasks": [{"local_id": "t1", "label": "Other", "start_segment_id": ids[0], "end_segment_id": ids[-1],
                           "summary": "chunk", "confidence": 0.5, "repeat_of": None,
                           "steps": [{"description": "a", "evidence_segment_ids": [ids[0]]},
                                     {"description": "b", "evidence_segment_ids": [ids[-1]]}]}]}

    prov = ScriptedProvider(fn)
    run, sc = run_llm(day1.segments, day1.window, cfg, prov)
    n_chunks = run.params["stats"]["chunks"]
    assert n_chunks == len(prov.calls) > 1
    assert len(run.tasks) < n_chunks                 # touching chunk tasks were merged
    starts = [t.start for t in run.tasks]
    assert starts == sorted(starts)
    assert len({t.task_id for t in run.tasks}) == len(run.tasks)


def test_failed_chunk_becomes_warning(day1, cfg):
    from workflow_recon.llm import LLMError

    def fn(system, user):
        raise LLMError("boom")
    run, _ = run_llm(day1.segments, day1.window, cfg, ScriptedProvider(fn))
    assert run.tasks == [] and run.params["stats"]["chunks_failed"] == 1
    assert "boom" in run.warnings[0]


def test_refuses_unredacted(day1, cfg):
    with pytest.raises(PrivacyError):
        run_llm(day1.clean.segments, day1.window, cfg, ScriptedProvider(lambda s, u: {"tasks": []}))


def test_leak_check_blocks_send(day1, cfg):
    segs = day1.segments.copy()
    segs.loc[segs.index[0], "title_norm"] = "contact me@example.com"
    prov = ScriptedProvider(lambda s, u: {"tasks": []})
    with pytest.raises(PrivacyError):
        run_llm(segs, day1.window, cfg, prov)
    assert prov.calls == []


def test_cache(day1, cfg, tmp_path):
    inner = ScriptedProvider(lambda s, u: {"tasks": []})
    prov = CachedProvider(inner, tmp_path / "llm", "v1")
    run_llm(day1.segments, day1.window, cfg, prov)
    run_llm(day1.segments, day1.window, cfg, prov)
    assert len(inner.calls) == 1 and prov.hits == 1


def test_history_reaches_prompt(day1, cfg):
    hist = [{"task_id": "B-2026-11-30-001", "label": "Internship application", "start": "2026-11-30T10:00:00-08:00",
             "summary": "Applied on Handshake.", "steps": ["Search Handshake"], "repeat_of": None}]
    prov = ScriptedProvider(lambda s, u: {"tasks": []})
    run_llm(day1.segments, day1.window, cfg, prov, history=hist)
    assert "B-2026-11-30-001" in prov.calls[0]


def test_prompt_and_schema():
    system, user = load_prompt("v1")
    assert "{{taxonomy}}" in system and "{{segments}}" in user
    schema = output_json_schema()
    task = schema["properties"]["tasks"]["items"]
    assert task["properties"]["label"]["enum"] == PREDICTABLE_LABELS
    assert "Private" not in PREDICTABLE_LABELS

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(schema)


def test_render_segments_format(day1):
    text = render_segments(day1.segments)
    first = text.splitlines()[0]
    assert re.match(r"^\[S0001\] \d\d:\d\d:\d\d–\d\d:\d\d:\d\d \(", first)
    assert "--- idle" in text


# ------------------------------------------------------------ Anthropic provider (no network)

class _FakeStream:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.msg


def _fake_client(msg, seen):
    def stream(**kw):
        seen.update(kw)
        return _FakeStream(msg)
    ns = SimpleNamespace(stream=stream)
    return SimpleNamespace(beta=SimpleNamespace(messages=ns), messages=ns)


def _msg(stop="end_turn", text='{"tasks": []}'):
    return SimpleNamespace(stop_reason=stop, stop_details=SimpleNamespace(category="cyber"),
                           content=[SimpleNamespace(type="text", text=text)], model="claude-opus-5-5",
                           usage=SimpleNamespace(input_tokens=12, output_tokens=3))


def test_anthropic_request_shape():
    seen = {}
    prov = AnthropicProvider("claude-opus-5-5", effort="high", client=_fake_client(_msg(), seen))
    resp = prov.generate_json("sys", "user", {"type": "object"}, 1000)
    assert resp.data == {"tasks": []} and resp.usage["input_tokens"] == 12
    assert seen["model"] == "claude-opus-5-5"
    assert seen["output_config"] == {"format": {"type": "json_schema", "schema": {"type": "object"}}, "effort": "high"}
    assert seen["betas"] == [FALLBACK_BETA] and seen["fallbacks"] == "default"
    assert "temperature" not in seen and "thinking" not in seen


def test_anthropic_refusal():
    prov = AnthropicProvider("claude-opus-5-5", client=_fake_client(_msg(stop="refusal"), {}))
    with pytest.raises(LLMRefusal):
        prov.generate_json("s", "u", {}, 10)


def test_missing_credentials_is_a_config_error():
    from workflow_recon.llm import LLMConfigError

    def stream(**kw):
        raise TypeError('"Could not resolve authentication method. Expected one of api_key..."')
    ns = SimpleNamespace(stream=stream)
    prov = AnthropicProvider("claude-opus-5-5", client=SimpleNamespace(beta=SimpleNamespace(messages=ns), messages=ns))
    with pytest.raises(LLMConfigError):
        prov.generate_json("s", "u", {}, 10)
