"""Fixes from the 2026-10-06 real-provider smoke (docs/notes/m1b-smoke-2026-10-06.md):
LLMAgent `extra` passthrough, `max_tokens` default, `length` notes, opt-in text tool fallback,
tolerant BeliefProbe parsing with candidate names from the context, and request shaping."""
import json
import logging

import pytest

from swarmlab.participants import LLMAgent
from swarmlab.participants.llm import parse_text_tool_calls
from swarmlab.probes import BeliefProbe, match_candidate, strip_reasoning
from swarmlab.providers.base import ChatMessage, ChatRequest
from swarmlab.providers.fake import FakeProvider, flaggame_reader
from swarmlab.providers.openai_compat import build_body
from swarmlab.tools import ToolSchema
from swarmlab.view import Part

from .helpers import _resp, llm_agent_experiment

PROBE_METRICS = ["belief.accuracy"]
TOOLS = [ToolSchema(name=n, description="", parameters={}).normalized()
         for n in ("guess", "end_turn", "read_board")]


# ---- fake scripts (resolved as fake:tests.test_smoke_fixes:<name>) ---------------------------------
def _reader_guess(request, rng) -> str:
    calls = flaggame_reader(request, rng).tool_calls
    return next((c.args["candidate"] for c in calls if c.name == "guess"), "A")


def script_text_calls(request, rng):
    """Writes its calls as text, the way a model without working native tool calls might."""
    if not request.tools:
        return _resp(request, '{"candidate": "A"}')
    if request.messages[-1].role == "user" and str(request.messages[-1].content).startswith("[tool results]"):
        return _resp(request, "done")
    return _resp(request, f'I will now call guess(candidate="{_reader_guess(request, rng)}") and then end_turn().')


def script_hermes_calls(request, rng):
    if not request.tools:
        return _resp(request, '{"candidate": "A"}')
    return _resp(request, '<think>hmm</think>\n<tool_call>\n{"name": "guess", "arguments": {"candidate": "B"}}'
                          '\n</tool_call>\n<tool_call>\n{"name": "end_turn", "arguments": {}}\n</tool_call>')


def script_length(request, rng):
    return _resp(request, "", finish="length")


def script_messy_probe(request, rng):
    """Reader for turns; probe answers wrapped in reasoning, prose and a fence, lower-case alias."""
    if request.tools:
        return flaggame_reader(request, rng)
    cand = json.loads(flaggame_reader(request, rng).text)["candidate"]
    return _resp(request, f'<think>compare crops...</think>\nSo:\n```json\n{{"Answer": "{cand.lower()}", '
                          f'"confidence": 0.7}}\n```')


# ---- LLMAgent ---------------------------------------------------------------------------------------
@pytest.fixture
def requests_seen(monkeypatch):
    seen: list = []
    original = FakeProvider.complete

    async def recording(self, request):
        seen.append(request)
        return await original(self, request)

    monkeypatch.setattr(FakeProvider, "complete", recording)
    return seen


def turns(run):
    return [e for e in run.events if e["type"] == "turn_ended"]


def test_max_tokens_default_is_2048():
    assert LLMAgent(model="fake:reader").max_tokens == 2048


def test_extra_reaches_turn_and_probe_requests(tmp_path, requests_seen):
    extra = {"chat_template_kwargs": {"enable_thinking": False}}
    exp = llm_agent_experiment(2, agent_kw={"extra": extra}, probes=[BeliefProbe()], metrics=PROBE_METRICS)
    run = exp.run(seed=1, max_rounds=1, out=tmp_path)
    assert requests_seen and all(r.extra == extra for r in requests_seen)
    assert any(not r.tools for r in requests_seen)  # probe requests carry it too
    assert run.spec.participants[0].params["extra"] == extra
    assert all(e["ok"] for e in run.events if e["type"] == "probe")


def test_extra_is_a_body_field_for_openai_compat():
    req = ChatRequest(model="hf:Qwen/Qwen3.5-9B:deepinfra", messages=[ChatMessage(role="user", content="hi")],
                      extra={"chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "low",
                             "extra_body": {"top_k": 20}})
    body = build_body(req)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["reasoning_effort"] == "low" and body["top_k"] == 20 and "extra_body" not in body


def test_turn_notes_finish_reasons_and_length(tmp_path):
    run = llm_agent_experiment(1, agent_kw={"model": "fake:tests.test_smoke_fixes:script_length"}).run(
        seed=1, max_rounds=1, out=tmp_path)
    (t,) = turns(run)
    assert t["yield_kind"] == "no_tool"
    assert t["usage"]["finish_reasons"] == ["length"] and t["usage"]["notes"] == ["length"]
    ok = llm_agent_experiment(1).run(seed=1, max_rounds=1, out=tmp_path / "ok")
    assert turns(ok)[0]["usage"]["notes"] == [] and turns(ok)[0]["usage"]["finish_reasons"]


def test_text_tool_fallback_off_by_default(tmp_path):
    run = llm_agent_experiment(2, agent_kw={"model": "fake:tests.test_smoke_fixes:script_text_calls"}).run(
        seed=1, max_rounds=1, out=tmp_path)
    assert {t["yield_kind"] for t in turns(run)} == {"no_tool"}
    assert run.score["n_guessed"] == 0


def test_text_tool_fallback_executes_written_calls(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="swarmlab.participants.llm")
    exp = llm_agent_experiment(2, agent_kw={"model": "fake:tests.test_smoke_fixes:script_text_calls",
                                            "text_tool_fallback": True})
    run = exp.run(seed=1, max_rounds=1, out=tmp_path)
    assert {t["yield_kind"] for t in turns(run)} == {"end_turn"}
    assert run.score["n_guessed"] == 2
    assert all("text_tool_fallback:2" in t["usage"]["notes"] for t in turns(run))
    assert "text_tool_fallback parsed 2 call(s)" in caplog.text
    called = [e["tool"] for e in run.events if e["type"] == "tool_called"]
    assert called == ["guess", "end_turn"] * 2


def test_text_tool_fallback_hermes_blocks(tmp_path):
    exp = llm_agent_experiment(1, agent_kw={"model": "fake:tests.test_smoke_fixes:script_hermes_calls",
                                            "text_tool_fallback": True})
    run = exp.run(seed=1, max_rounds=1, out=tmp_path)
    assert turns(run)[0]["yield_kind"] == "end_turn"
    guess = next(e for e in run.events if e["type"] == "tool_called")
    assert guess["tool"] == "guess" and guess["args"] == {"candidate": "B"}


@pytest.mark.parametrize("text, expected", [
    ('guess(candidate="B")', [("guess", {"candidate": "B"})]),
    ("Let me guess(candidate='C'), then end_turn()", [("guess", {"candidate": "C"}), ("end_turn", {})]),
    ('read_board(limit=5, channel="main")', [("read_board", {"limit": 5, "channel": "main"})]),
    ('<tool_call>{"name": "guess", "arguments": {"candidate": "D"}}</tool_call>', [("guess", {"candidate": "D"})]),
    ('```json\n[{"name": "guess", "args": {"candidate": "E"}}]\n```', [("guess", {"candidate": "E"})]),
    ('<think>maybe guess(candidate="X")</think> guess(candidate="F")', [("guess", {"candidate": "F"})]),
    ("I think the answer is B.", []),
    ('post(text="hi")', []),  # not an offered tool
    ('guess("B")', []),  # positional args are not accepted
    ('{"candidate": "B"}', []),  # a bare object is not a call
])
def test_parse_text_tool_calls(text, expected):
    assert parse_text_tool_calls(text, TOOLS) == expected


# ---- BeliefProbe ------------------------------------------------------------------------------------
NAMES = list("ABCDEFGH")


@pytest.mark.parametrize("text, cands, parsed", [
    ('{"candidate": "C", "confidence": 0.85}', None, {"candidate": "C", "confidence": 0.85}),
    ('<think>{"candidate": "A"}</think>{"candidate": "B"}', None, {"candidate": "B"}),
    ('reasoning only, then </think>\n{"candidate": "B"}', None, {"candidate": "B"}),
    ('Here you go:\n```json\n{"guess": "c", "confidence": 1}\n```', NAMES, {"candidate": "C", "confidence": 1.0}),
    ('{"Answer": "Candidate G"}', NAMES, {"candidate": "G"}),
    ('{"flag": "h."}', NAMES, {"candidate": "H"}),
    ('{"candidate": "Z"}', NAMES, {"candidate": "Z", "unknown_candidate": True}),
    ('\n\n```json\n{"candidate": "C", "', NAMES, {"candidate": "C", "partial": True}),  # smoke: truncated
    ('{"candidate": 3}', None, {"candidate": "3"}),
])
def test_belief_parse_tolerant(text, cands, parsed):
    ok, got = BeliefProbe().parse(text, cands)
    assert ok and got == parsed


@pytest.mark.parametrize("text", ["", "<think>still thinking about G and", "no json here", '{"confidence": 1}'])
def test_belief_parse_failures(text):
    ok, got = BeliefProbe().parse(text, NAMES)
    assert not ok and "error" in got


def test_strip_reasoning_and_match_candidate():
    assert strip_reasoning("<think>a</think>b<think>c") == "b"
    assert strip_reasoning(None) == ""
    assert match_candidate(" `d` ", NAMES) == "D" and match_candidate("flag: e", NAMES) == "E"
    assert match_candidate("D", None) is None and match_candidate("DD", NAMES) is None


def test_candidates_from_context_reads_the_latest_flaggame_observation():
    obs = "Candidate flags:\n\nA:\nrr\nrr\n\nB:\ngg\ngg\n\nYour crop:\nr"
    ctx = [ChatMessage(role="system", content="sys"),
           ChatMessage(role="user", content=[Part(type="text", text="Round 1."), Part(type="text", text=obs)]),
           ChatMessage(role="user", content=[Part(type="text", text="Outcomes of your actions last round:\n- x")])]
    assert BeliefProbe().candidates_from_context(ctx) == ["A", "B"]
    assert BeliefProbe().candidates_from_context(ctx[:1]) is None
    assert BeliefProbe().candidates_from_context([ChatMessage(role="user", content="plain")]) is None


def test_runner_passes_candidate_names_to_parse(tmp_path):
    exp = llm_agent_experiment(3, agent_kw={"model": "fake:tests.test_smoke_fixes:script_messy_probe"},
                               probes=[BeliefProbe()], metrics=PROBE_METRICS)
    run = exp.run(seed=1, max_rounds=2, out=tmp_path)
    probes = [e for e in run.events if e["type"] == "probe"]
    assert len(probes) == 6 and all(e["ok"] for e in probes)
    assert all(e["parsed"]["candidate"] in NAMES and e["parsed"]["confidence"] == 0.7 for e in probes)
