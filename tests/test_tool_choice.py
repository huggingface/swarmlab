"""`ChatRequest.tool_choice`: hashing, adapter bodies, validation, the fake provider, LLMAgent."""
import asyncio
import json

import httpx
import pytest

from swarmlab.participants import LLMAgent
from swarmlab.providers import ToolChoice
from swarmlab.providers.anthropic import build_kwargs
from swarmlab.providers.base import ChatMessage, ChatRequest, request_bytes, request_hash
from swarmlab.providers.fake import FakeProvider, parse_listing
from swarmlab.providers.openai_compat import build_body
from swarmlab.spec import canonical_json

from .helpers import _resp, logical
from .test_providers import GUESS, LOOSE, anthropic_message, compat, completion, req, stub_provider

FORCE = ToolChoice(name="guess")


# ---- hashing ---------------------------------------------------------------------------------------
def test_unset_tool_choice_leaves_the_hash_unchanged():
    r = req(tools=[GUESS])
    legacy = {k: v for k, v in r.model_dump(mode="json").items() if k not in ("attempt", "tool_choice")}
    assert request_bytes(r) == canonical_json(legacy).encode()  # pre-`tool_choice` hashes hold
    forced = req(tools=[GUESS], tool_choice=FORCE)
    assert request_hash(forced) != request_hash(r)
    assert json.loads(request_bytes(forced))["tool_choice"] == {"name": "guess"}
    again = ChatRequest.model_validate_json(forced.model_dump_json())
    assert again.tool_choice == FORCE and request_hash(again) == request_hash(forced)


# ---- adapters --------------------------------------------------------------------------------------
def test_anthropic_sends_a_named_tool_choice():
    assert build_kwargs(req(tools=[GUESS, LOOSE]))["tool_choice"] == {"type": "auto"}
    kw = build_kwargs(req(tools=[GUESS, LOOSE], tool_choice=FORCE))
    assert kw["tool_choice"] == {"type": "tool", "name": "guess"}
    assert [t["name"] for t in kw["tools"]] == ["guess", "read_board"]
    p = stub_provider([anthropic_message()])
    asyncio.run(p.complete(req(tools=[GUESS], tool_choice=FORCE)))
    assert p._client.messages.kwargs[0]["tool_choice"] == {"type": "tool", "name": "guess"}


def test_anthropic_refuses_tool_choice_with_thinking():
    with pytest.raises(ValueError, match="thinking"):
        build_kwargs(req(tools=[GUESS], tool_choice=FORCE, thinking_budget=2048))
    # a model this adapter sends no thinking to is unaffected
    kw = build_kwargs(req(model="anthropic:claude-x", tools=[GUESS], tool_choice=FORCE,
                          thinking_budget=2048))
    assert kw["tool_choice"]["type"] == "tool" and "thinking" not in kw


def test_openai_compat_sends_a_named_tool_choice():
    assert build_body(req(model="hf:m", tools=[GUESS]))["tool_choice"] == "auto"
    body = build_body(req(model="hf:m", tools=[GUESS, LOOSE], tool_choice=FORCE))
    assert body["tool_choice"] == {"type": "function", "function": {"name": "guess"}}
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=completion({"content": "", "tool_calls": [
            {"id": "c1", "function": {"name": "guess", "arguments": '{"candidate": "A"}'}}]},
            finish="tool_calls"))

    resp = asyncio.run(compat(handler).complete(req(model="hf:m", tools=[GUESS], tool_choice=FORCE)))
    assert sent[0]["tool_choice"] == {"type": "function", "function": {"name": "guess"}}
    assert resp.tool_calls[0].name == "guess"


# ---- validation ------------------------------------------------------------------------------------
def test_unknown_tool_name_is_a_value_error():
    with pytest.raises(ValueError, match="not among the request's tools"):
        req(tools=[LOOSE], tool_choice=FORCE)
    with pytest.raises(ValueError, match="not among"):
        req(tool_choice=FORCE)  # no tools at all
    # a request changed after construction is checked again by every adapter
    r = req(tools=[GUESS], tool_choice=FORCE)
    r.tools = [LOOSE]
    for build in (build_kwargs, build_body):
        with pytest.raises(ValueError, match="not among"):
            build(r)
    with pytest.raises(ValueError, match="not among"):
        asyncio.run(FakeProvider().complete(r))


def test_json_protocol_cannot_force_a_tool():
    with pytest.raises(ValueError, match="tool_choice requires tool_protocol native"):
        req(tools=[GUESS], tool_protocol="json", tool_choice=FORCE)
    r = req(tools=[GUESS], tool_choice=FORCE)
    r.tool_protocol = "json"
    for build in (build_kwargs, build_body):
        with pytest.raises(ValueError, match="tool_protocol native"):
            build(r)


def test_tool_choice_in_extra_too_is_a_conflict():
    with pytest.raises(ValueError, match="also in extra"):
        req(tools=[GUESS], tool_choice=FORCE, extra={"tool_choice": "required"})
    # without the field, extra still passes through unchanged
    raw = {"type": "function", "function": {"name": "guess"}}
    assert build_body(req(model="hf:m", tools=[GUESS], extra={"tool_choice": raw}))["tool_choice"] == raw


# ---- fake provider ---------------------------------------------------------------------------------
def script_answers_in_text(request, rng):
    return _resp(request, json.dumps({"candidate": "A"}))


def script_plain_text(request, rng):
    return _resp(request, "thinking it over")


def script_reads_then_guesses(request, rng):
    return _resp(request, "", calls=[("read_board", {"limit": 5}), ("guess", {"candidate": "B"})])


def script_guesses(request, rng):
    return _resp(request, "", calls=[("guess", {"candidate": "C"}), ("read_board", {})])


def _fake(script, **kw):
    msgs = [ChatMessage(role="user", content="go")]
    request = ChatRequest(model="fake:x", messages=msgs, tools=[GUESS, LOOSE], **kw)
    return asyncio.run(FakeProvider(script=f"tests.test_tool_choice:{script}").complete(request))


def test_fake_provider_honours_tool_choice():
    unforced = _fake("script_answers_in_text")
    assert unforced.tool_calls == [] and unforced.finish_reason == "end_turn"
    # a JSON-object text answer becomes the forced call's args
    r = _fake("script_answers_in_text", tool_choice=FORCE)
    assert [(c.name, c.args) for c in r.tool_calls] == [("guess", {"candidate": "A"})]
    assert r.text == "" and r.finish_reason == "tool_use" and r.tool_calls[0].call_id.startswith("fake_")
    assert r == _fake("script_answers_in_text", tool_choice=FORCE)  # deterministic
    # other text stays; the call gets empty args
    r = _fake("script_plain_text", tool_choice=FORCE)
    assert r.text == "thinking it over" and [(c.name, c.args) for c in r.tool_calls] == [("guess", {})]
    # the named call is moved to the front, other calls kept
    r = _fake("script_reads_then_guesses", tool_choice=FORCE)
    assert [c.name for c in r.tool_calls] == ["guess", "read_board"]
    assert r.tool_calls[0].args == {"candidate": "B"}
    # already first: unchanged
    assert _fake("script_guesses", tool_choice=FORCE).tool_calls == _fake("script_guesses").tool_calls


def test_fake_provider_forces_a_tool_the_script_did_not_call():
    r = _fake("script_reads_then_guesses", tool_choice=ToolChoice(name="read_board"))
    assert [c.name for c in r.tool_calls] == ["read_board", "guess"]


# ---- LLMAgent round trip -----------------------------------------------------------------------------
def script_names_a_candidate(request, rng):
    """Answers in text only: `{"candidate": <first listed>}` (a forced `guess` carries it)."""
    from swarmlab.providers.base import text_of
    from swarmlab.world.flaggame import PREAMBLE

    listing = next(text_of(m.content) for m in reversed(request.messages)
                   if m.role == "user" and PREAMBLE in text_of(m.content))
    return _resp(request, json.dumps({"candidate": min(parse_listing(listing)[0])}))


class ForceGuessAgent(LLMAgent):
    """Forces `guess` on the first model call of each turn (a `_request` override)."""

    def _request(self, tools, attempt=0):
        req = super()._request(tools, attempt)
        turn = getattr(self, "_turn", None)
        if any(t.name == "guess" for t in tools) and (turn is None or turn.calls == 0):
            req = req.model_copy(update={"tool_choice": ToolChoice(name="guess")})
        return req


def test_llm_agent_subclass_forces_a_tool_in_a_fake_run(tmp_path, monkeypatch):
    from .test_llm_agent import turns
    from .test_llm_loop_seam import _subclass_experiment

    seen = []
    original = FakeProvider.complete

    async def recording(self, request):
        seen.append(request)
        return await original(self, request)

    monkeypatch.setattr(FakeProvider, "complete", recording)
    agent = ForceGuessAgent(model="fake:tests.test_tool_choice:script_names_a_candidate")
    run = _subclass_experiment(agent).run(seed=1, max_rounds=2, out=tmp_path)
    assert run.end_reason == "max_rounds" and run.score["n_guessed"] == 2
    forced = [r for r in seen if r.tool_choice is not None]
    assert len(forced) == 4 and all(r.tool_choice.name == "guess" for r in forced)
    called = [(e["tool"], e["args"]) for e in run.events if e["type"] == "tool_called"]
    assert [name for name, _ in called] == ["guess"] * 4
    assert all(set(args) == {"candidate"} for _, args in called)
    assert all(t["usage"]["calls"] == 1 for t in turns(run))
    again = _subclass_experiment(agent).run(seed=1, max_rounds=2, out=tmp_path / "again")
    assert logical(again, exclude=("seq", "ts", "run")) == logical(run, exclude=("seq", "ts", "run"))


def test_llm_agent_json_protocol_with_tool_choice_errors_the_turn(tmp_path):
    from .test_llm_loop_seam import _subclass_experiment

    agent = ForceGuessAgent(model="fake:tests.test_tool_choice:script_names_a_candidate",
                            tool_protocol="json")
    run = _subclass_experiment(agent, n=1).run(seed=1, max_rounds=1, out=tmp_path)
    s = run.summary()  # never silently sent as auto: the turn errors and the run says why
    assert s["turns_errored"] == 1 and s["health"] == "degraded"
    assert s["first_error"].startswith("ValueError: tool_choice requires tool_protocol native")
