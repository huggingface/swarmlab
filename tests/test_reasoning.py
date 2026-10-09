"""Reasoning traces: Anthropic thinking blocks and OpenAI-compatible reasoning text are captured,
cached, replayed, exported and shown, and thinking blocks go back unchanged in tool loops
(stub Anthropic clients and `httpx.MockTransport`; no paid calls)."""
import json
from types import SimpleNamespace

import httpx
import pyarrow.parquet as pq
import pytest

from swarmlab import Board, Experiment, Run
from swarmlab.participants import LLMAgent
from swarmlab.providers.anthropic import AnthropicProvider, build_kwargs
from swarmlab.providers.base import ChatMessage, ChatRequest, request_bytes
from swarmlab.providers.openai_compat import build_body
from swarmlab.tools import ToolCall
from swarmlab.viewer.build import build, collect
from swarmlab.world.flaggame import FlagGame

from .test_providers import anthropic_message, compat, completion, req, stub_provider
from .test_recovery import _truncate_mid_round


def thinking(text, sig):
    return SimpleNamespace(type="thinking", thinking=text, signature=sig)


def usage(thinking_tokens=0):
    return SimpleNamespace(input_tokens=100, output_tokens=60, cache_read_input_tokens=0,
                           cache_creation_input_tokens=0,
                           output_tokens_details=SimpleNamespace(thinking_tokens=thinking_tokens))


# ---- adapters ------------------------------------------------------------------------------------------
async def test_anthropic_summarized_thinking_is_captured_in_order():
    msg = anthropic_message(content=[
        thinking("The crop has a red band.", "sig-1"),
        SimpleNamespace(type="text", text="Guessing A.", citations=None),
        SimpleNamespace(type="tool_use", id="toolu_1", name="guess", input={"candidate": "A"}),
    ], stop_reason="tool_use")
    msg.usage = usage(thinking_tokens=42)
    resp = await stub_provider([msg]).complete(req())
    assert resp.reasoning == "The crop has a red band." and resp.reasoning_kind == "summary"
    assert resp.reasoning_redacted == 0 and resp.usage.reasoning_tokens == 42
    assert resp.text == "Guessing A." and resp.tool_calls[0].name == "guess"
    assert resp.provider_content == [
        {"type": "thinking", "thinking": "The crop has a red band.", "signature": "sig-1"},
        {"type": "text", "text": "Guessing A."},
        {"type": "tool_use", "id": "toolu_1", "name": "guess", "input": {"candidate": "A"}}]


async def test_anthropic_omitted_and_redacted_thinking():
    msg = anthropic_message(content=[
        thinking("", "sig-1"), SimpleNamespace(type="redacted_thinking", data="enc-1"),
        thinking("", "sig-2"), SimpleNamespace(type="text", text="ok", citations=None)])
    resp = await stub_provider([msg]).complete(req())
    assert resp.reasoning == "" and resp.reasoning_kind == "omitted" and resp.reasoning_redacted == 1
    assert [b["type"] for b in resp.provider_content] == [
        "thinking", "redacted_thinking", "thinking", "text"]
    assert resp.provider_content[1] == {"type": "redacted_thinking", "data": "enc-1"}


async def test_no_thinking_leaves_every_field_unset():
    resp = await stub_provider([anthropic_message()]).complete(req())
    assert resp.reasoning is None and resp.reasoning_kind is None
    assert resp.provider_content is None and resp.usage.reasoning_tokens == 0


def test_provider_content_is_sent_verbatim_and_only_when_set():
    blocks = [{"type": "thinking", "thinking": "", "signature": "s1"},
              {"type": "tool_use", "id": "t1", "name": "read_board", "input": {}},
              {"type": "thinking", "thinking": "", "signature": "s2"},
              {"type": "tool_use", "id": "t2", "name": "end_turn", "input": {}}]
    calls = [ToolCall(call_id="t1", name="read_board", args={}),
             ToolCall(call_id="t2", name="end_turn", args={})]
    msgs = [ChatMessage(role="user", content="Round 1."),
            ChatMessage(role="assistant", content="", tool_calls=calls, provider_content=blocks),
            ChatMessage(role="tool", content="{}", tool_call_id="t1"),
            ChatMessage(role="tool", content="{}", tool_call_id="t2")]
    kw = build_kwargs(ChatRequest(model="anthropic:claude-opus-5-5", messages=msgs))
    assert kw["messages"][1] == {"role": "assistant", "content": blocks}  # interleaving kept
    plain = build_kwargs(ChatRequest(model="anthropic:claude-opus-5-5", messages=[
        msgs[0], msgs[1].model_copy(update={"provider_content": None}), *msgs[2:]]))
    assert [b["type"] for b in plain["messages"][1]["content"]] == ["tool_use", "tool_use"]
    # OpenAI-compatible bodies never carry it
    assert build_body(ChatRequest(model="hf:m", messages=msgs)) == build_body(
        ChatRequest(model="hf:m", messages=[m.model_copy(update={"provider_content": None})
                                            for m in msgs]))


def test_requests_without_reasoning_hash_as_before():
    r = req(messages=[ChatMessage(role="user", content="hi"),
                      ChatMessage(role="assistant", content="yo")])
    data = json.loads(request_bytes(r))
    assert all("provider_content" not in m for m in data["messages"])
    assert ChatMessage(role="assistant", content="x").model_dump() == {
        "role": "assistant", "content": "x", "tool_calls": None, "tool_call_id": None}


@pytest.mark.parametrize("key", ["reasoning_content", "reasoning"])
async def test_openai_reasoning_text_is_captured(key):
    def handler(request):
        return httpx.Response(200, json=completion(
            {"content": "Answer.", key: "First, the crop rows..."}, finish="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 30,
                   "completion_tokens_details": {"reasoning_tokens": 20}}))

    resp = await compat(handler).complete(req(model="hf:m"))
    assert resp.reasoning == "First, the crop rows..." and resp.reasoning_kind == "text"
    assert resp.provider_content is None and resp.usage.reasoning_tokens == 20

    def plain(request):
        return httpx.Response(200, json=completion({"content": "Answer.", key: None}))

    assert (await compat(plain).complete(req(model="hf:m"))).reasoning is None


# ---- a run on a stubbed Anthropic client -------------------------------------------------------------------
class ThinkingClaude:
    """`messages.create` stub: a round's first call thinks and reads the board, the call after
    the tool results thinks again and ends the turn. Records every request's kwargs."""

    def __init__(self):
        self.kwargs: list[dict] = []

    async def create(self, **kw):
        self.kwargs.append(kw)
        n = len(self.kwargs)
        # a new round's text joins the previous tool results in one user message: look at the end
        after_tools = kw["messages"][-1]["content"][-1].get("type") == "tool_result"
        if after_tools:
            content = [thinking(f"Board read; done (call {n}).", f"sig-{n}"),
                       SimpleNamespace(type="tool_use", id=f"toolu_{n}", name="end_turn", input={})]
        else:
            content = [thinking(f"I should read the board first (call {n}).", f"sig-{n}"),
                       SimpleNamespace(type="text", text="Let me look.", citations=None),
                       SimpleNamespace(type="tool_use", id=f"toolu_{n}", name="read_board",
                                       input={"limit": 10})]
        msg = anthropic_message(content=content, stop_reason="tool_use")
        msg.usage = usage(thinking_tokens=25)
        return msg


def claude_experiment(stub, **agent_kw):
    p = AnthropicProvider()
    p._client = SimpleNamespace(messages=stub)
    return Experiment(name="think", world=FlagGame(),
                      participants=[LLMAgent(model="anthropic:claude-haiku-4-5", **agent_kw)],
                      medium=Board(topology="broadcast"), providers={"anthropic": p})


def assistant_turns(kw):
    return [m for m in kw["messages"] if m["role"] == "assistant"]


def test_thinking_goes_back_unchanged_in_the_tool_loop(tmp_path):
    stub = ThinkingClaude()
    run = claude_experiment(stub).run(seed=1, max_rounds=2, out=tmp_path)
    assert run.end_reason == "max_rounds" and len(stub.kwargs) == 4
    first, second, third = stub.kwargs[0], stub.kwargs[1], stub.kwargs[2]
    assert assistant_turns(first) == []
    # the second call of round 1 replays the first response's blocks verbatim, thinking first
    (echo,) = assistant_turns(second)
    assert echo["content"] == [
        {"type": "thinking", "thinking": "I should read the board first (call 1).",
         "signature": "sig-1"},
        {"type": "text", "text": "Let me look."},
        {"type": "tool_use", "id": "toolu_1", "name": "read_board", "input": {"limit": 10}}]
    # full memory is append-only: round 2 still carries round 1's thinking, in order
    sigs = [b.get("signature") for m in assistant_turns(third) for b in m["content"]
            if b["type"] == "thinking"]
    assert sigs == ["sig-1", "sig-2"]
    t = next(e for e in run.events if e["type"] == "turn_ended")
    assert t["usage"]["reasoning_tokens"] == 50  # 2 calls x 25 thinking tokens


def test_window_trim_drops_all_kept_thinking(tmp_path):
    stub = ThinkingClaude()
    claude_experiment(stub, memory="window", window_rounds=2).run(seed=1, max_rounds=3, out=tmp_path)
    # round 2 starts with round 1 intact (nothing trimmed yet): its thinking is replayed
    r2 = stub.kwargs[2]
    assert any(b["type"] == "thinking" for m in assistant_turns(r2) for b in m["content"])
    # round 3 drops round 1, so round 2's blocks (bound to round 1) are no longer sent
    r3_first, r3_second = stub.kwargs[4], stub.kwargs[5]
    assert assistant_turns(r3_first) and not any(
        b["type"] == "thinking" for m in assistant_turns(r3_first) for b in m["content"])
    # within round 3 the new blocks go back as usual
    assert assistant_turns(r3_second)[-1]["content"][0]["signature"] == "sig-5"


def test_reasoning_survives_cache_replay_and_resume(tmp_path):
    stub = ThinkingClaude()
    exp = claude_experiment(stub)
    run = exp.run(seed=1, max_rounds=3, out=tmp_path)
    made = len(stub.kwargs)
    responses = [e for e in run.events_all if e.type == "inference_response"]
    cached = json.loads((run.dir / "blobs" / "cache" / next(
        e.request_hash for e in run.events_all if e.type == "inference_attempt")).read_text())
    assert cached["reasoning"] == "I should read the board first (call 1)."
    assert cached["provider_content"][0]["signature"] == "sig-1"
    loaded = Run.load(run.dir)  # full replay
    assert len(stub.kwargs) == made and loaded.score == run.score
    _truncate_mid_round(run.dir, 2)
    resumed = Run(run.dir, experiment=exp).resume()
    assert len(stub.kwargs) == made  # every re-made request (thinking echoed) hit the cache
    again = [e for e in resumed.events_all if e.type == "inference_response" and e.round >= 2]
    assert again and all(e.cached for e in again if e.round == 2)
    assert len(responses) == made


def test_export_and_viewer_show_reasoning(tmp_path):
    stub = ThinkingClaude()
    run = claude_experiment(stub).run(seed=1, max_rounds=1, out=tmp_path / "runs")
    out = run.export(tmp_path / "export")
    rows = pq.read_table(out / "tables" / "inference.parquet").to_pylist()
    assert [(r["reasoning"], r["reasoning_kind"], r["reasoning_redacted"]) for r in rows] == [
        ("I should read the board first (call 1).", "summary", 0),
        ("Board read; done (call 2).", "summary", 0)]
    assert all(r["reasoning_tokens"] == 25 for r in rows)
    lines = [json.loads(x) for x in (out / "sessions" / "a000.jsonl").read_text().splitlines()]
    assistants = [e["message"] for e in lines[1:] if e["message"]["role"] == "assistant"]
    assert assistants[0]["content"] == [
        {"type": "thinking", "thinking": "I should read the board first (call 1).",
         "thinkingSignature": "sig-1"},
        {"type": "text", "text": "Let me look."},
        {"type": "toolCall", "id": "toolu_1", "name": "read_board", "arguments": {"limit": 10}}]
    assert assistants[1]["content"][0]["thinkingSignature"] == "sig-2"
    calls = collect(run.dir)["calls"]
    assert [(c["agent"], c["reasoning"], c["reasoning_kind"]) for c in calls] == [
        ("a000", "I should read the board first (call 1).", "summary"),
        ("a000", "Board read; done (call 2).", "summary")]
    page = build(run.dir).read_text()
    assert "Model calls" in page and "read the board first" in page


def test_large_reasoning_is_referenced_by_hash(tmp_path, monkeypatch):
    from swarmlab import export

    monkeypatch.setattr(export, "INLINE_LIMIT", 20)
    stub = ThinkingClaude()
    run = claude_experiment(stub).run(seed=1, max_rounds=1, out=tmp_path / "runs")
    out = run.export(tmp_path / "export")
    rows = pq.read_table(out / "tables" / "inference.parquet").to_pylist()
    for r in rows:
        assert r["reasoning"] == f"sha256:{r['response_hash']}"
        assert (out / "raw" / "blobs" / r["response_hash"][:2] / r["response_hash"]).exists()


async def test_context_limit_trim_drops_all_kept_thinking():
    agent = LLMAgent(model="anthropic:claude-opus-5-5", context_limit_tokens=450)
    agent._estimator = SimpleNamespace(
        estimate_prompt_tokens=lambda r: 100 * sum(1 for m in r.messages if m.role != "system"))
    agent.system = "sys"
    block = [{"type": "thinking", "thinking": "", "signature": "s"}, {"type": "text", "text": "a"}]
    agent.rounds = [{"round": r, "messages": [
        ChatMessage(role="user", content=f"Round {r}.").model_dump(mode="json"),
        ChatMessage(role="assistant", content="a", provider_content=block).model_dump(mode="json")]}
        for r in (1, 2, 3)]
    await agent._enforce_context_limit([], tools=None)
    assert [e["round"] for e in agent.rounds] == [2, 3]
    assert all("provider_content" not in m for e in agent.rounds for m in e["messages"])
    # nothing trimmed: the blocks stay
    agent.rounds = agent.rounds[-1:]
    agent.rounds[0]["messages"][1]["provider_content"] = block
    await agent._enforce_context_limit([], tools=None)
    assert agent.rounds[0]["messages"][1]["provider_content"] == block
