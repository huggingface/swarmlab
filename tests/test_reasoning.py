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

    def __init__(self, pad: int = 0, pad_on: tuple[int, ...] = ()):
        self.kwargs: list[dict] = []
        self.pad, self.pad_on = pad, pad_on  # a `pad`-char signature on the calls in `pad_on`

    def sig(self, n):
        return f"sig-{n}" + ("s" * self.pad if n in self.pad_on else "")

    async def create(self, **kw):
        self.kwargs.append(kw)
        n = len(self.kwargs)
        # a new round's text joins the previous tool results in one user message: look at the end
        after_tools = kw["messages"][-1]["content"][-1].get("type") == "tool_result"
        if after_tools:
            content = [thinking(f"Board read; done (call {n}).", self.sig(n)),
                       SimpleNamespace(type="tool_use", id=f"toolu_{n}", name="end_turn", input={})]
        else:
            content = [thinking(f"I should read the board first (call {n}).", self.sig(n)),
                       SimpleNamespace(type="text", text="Let me look.", citations=None),
                       SimpleNamespace(type="tool_use", id=f"toolu_{n}", name="read_board",
                                       input={"limit": 10})]
        msg = anthropic_message(content=content, stop_reason="tool_use")
        msg.usage = usage(thinking_tokens=25)
        return msg


def claude_experiment(stub, prefix="anthropic", **agent_kw):
    p = AnthropicProvider()
    p._client = SimpleNamespace(messages=stub)
    return Experiment(name="think", world=FlagGame(),
                      participants=[LLMAgent(model=f"{prefix}:claude-haiku-4-5", **agent_kw)],
                      medium=Board(topology="broadcast"), providers={prefix: p})


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
    # default: thinking lives within one swarm turn; round 2 starts without round 1's blocks
    assert thinking_sigs(third) == [] and assistant_turns(third)  # text and tool calls stay
    assert thinking_sigs(stub.kwargs[3]) == ["sig-3"]  # round 2's own loop still returns them
    t = next(e for e in run.events if e["type"] == "turn_ended")
    assert t["usage"]["reasoning_tokens"] == 50  # 2 calls x 25 thinking tokens


def test_carry_thinking_keeps_earlier_turns_blocks(tmp_path):
    stub = ThinkingClaude()
    claude_experiment(stub, carry_thinking=True).run(seed=1, max_rounds=2, out=tmp_path)
    assert thinking_sigs(stub.kwargs[2]) == ["sig-1", "sig-2"]  # full memory: append-only
    assert thinking_sigs(stub.kwargs[3]) == ["sig-1", "sig-2", "sig-3"]


def test_carry_thinking_param_is_opt_in():
    assert "carry_thinking" not in LLMAgent(model="fake:reader").params
    assert LLMAgent(model="fake:reader", carry_thinking=True).params["carry_thinking"] is True
    with pytest.raises(ValueError, match="carry_thinking"):
        LLMAgent(model="fake:reader", carry_thinking="no")


def test_window_trim_drops_all_kept_thinking(tmp_path):
    stub = ThinkingClaude()
    claude_experiment(stub, memory="window", window_rounds=2, carry_thinking=True).run(
        seed=1, max_rounds=3, out=tmp_path)
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


# ---- kept thinking is dropped when the request's prefix changes ----------------------------------------
def system_text(kw):  # the provider caches by default: system goes as one marked text block
    return "".join(b["text"] for b in kw["system"])


def thinking_sigs(kw):
    return [b.get("signature") for m in assistant_turns(kw) for b in m["content"]
            if b["type"] == "thinking"]


def test_system_prompt_reconfigure_drops_kept_thinking(tmp_path):
    from swarmlab.interventions import Reconfigure

    stub = ThinkingClaude()
    exp = claude_experiment(stub, carry_thinking=True)
    exp.interventions = [Reconfigure(agents=["a000"], settings={"system_prompt": "A new system prompt."},
                                     at_round=1)]
    run = exp.run(seed=1, max_rounds=2, out=tmp_path)
    r2_first, r2_second = stub.kwargs[2], stub.kwargs[3]
    assert system_text(r2_first) == "A new system prompt."
    assert thinking_sigs(r2_first) == []  # sig-1 / sig-2 were bound to the old prompt
    assert thinking_sigs(r2_second) == ["sig-3"]  # new blocks go back as usual
    # what the responses recorded is untouched (analysis keeps it)
    out = run.export(tmp_path / "export")
    rows = pq.read_table(out / "tables" / "inference.parquet").to_pylist()
    assert [r["reasoning_kind"] for r in rows] == ["summary"] * 4


def kept_agent(binding_tools):
    agent = LLMAgent(model="anthropic:claude-opus-5-5", carry_thinking=True)
    agent.system = "sys"
    block = [{"type": "thinking", "thinking": "", "signature": "s"}, {"type": "text", "text": "a"}]
    agent.rounds = [{"round": 1, "messages": [
        ChatMessage(role="user", content="Round 1.").model_dump(mode="json"),
        ChatMessage(role="assistant", content="a", provider_content=block).model_dump(mode="json")]}]
    agent._check_reasoning_binding(binding_tools)
    agent.reasoning_binding = agent._request_prefix  # as if the response came from that request
    return agent


def kept(agent):
    return [m for e in agent.rounds for m in e["messages"] if "provider_content" in m]


def test_tool_schema_change_drops_kept_thinking():
    from swarmlab.tools import ToolSchema

    a = ToolSchema(name="guess", description="Guess.", parameters={"type": "object", "properties": {}})
    b = ToolSchema(name="guess", description="Guess the flag.", parameters={"type": "object", "properties": {}})
    agent = kept_agent([a])
    agent._check_reasoning_binding([a])  # same prefix: kept
    assert kept(agent)
    agent._check_reasoning_binding([a, b.model_copy(update={"name": "post"})])  # a tool added
    assert not kept(agent) and agent.reasoning_binding is None
    agent = kept_agent([a])
    agent._check_reasoning_binding([b])  # same name, changed schema
    assert not kept(agent)


def test_model_change_alone_keeps_thinking():
    agent = kept_agent([])
    agent.reconfigure(model="anthropic:claude-fable-5-1")
    agent._check_reasoning_binding([])
    assert kept(agent)  # the API drops what the new model cannot read; we do not strip


def test_old_snapshots_without_a_binding_keep_working():
    agent = kept_agent([])
    del agent.__dict__["reasoning_binding"]  # restored from a snapshot written before the field
    agent._check_reasoning_binding([])
    assert kept(agent)


# ---- cleanup runs before the context-limit estimate; disk traces are never touched -----------------------
# `research:` has no preset, so the agent estimates with ceil(chars / 4) of the request JSON, which
# counts thinking signatures. Round 1's last response (call 2, never re-sent within round 1)
# carries a 60k-char signature: ~15k tokens if round 2 sends it, against a 10k limit.
PAD = 60_000
LIMIT = 10_000


def assert_round1_traces_on_disk(run):
    attempts = [e for e in run.events_all if e.type == "inference_attempt" and e.round == 1]
    cached = [json.loads((run.dir / "blobs" / "cache" / a.request_hash).read_text()) for a in attempts]
    assert [len(c["provider_content"][0]["signature"]) for c in cached] == [5, PAD + 5]
    assert all(c["reasoning_kind"] == "summary" for c in cached)  # recorded as returned


def test_default_strips_earlier_turns_thinking_before_the_estimate(tmp_path):
    stub = ThinkingClaude(pad=PAD, pad_on=(2,))
    run = claude_experiment(stub, prefix="research", context_limit_tokens=LIMIT,
                            overflow="fail_turn").run(seed=1, max_rounds=2, out=tmp_path)
    turns = [e for e in run.events if e["type"] == "turn_ended"]
    assert [t["yield_kind"] for t in turns] == ["end_turn", "end_turn"]  # no context_limit error
    assert not [e for e in run.events if e["type"] == "overflow"]
    assert thinking_sigs(stub.kwargs[2]) == []
    assert_round1_traces_on_disk(run)


def test_opt_in_prompt_change_strips_before_the_estimate(tmp_path):
    from swarmlab.interventions import Reconfigure

    stub = ThinkingClaude(pad=PAD, pad_on=(2,))
    exp = claude_experiment(stub, prefix="research", carry_thinking=True,
                            context_limit_tokens=LIMIT, overflow="fail_turn")
    exp.interventions = [Reconfigure(agents=["a000"], settings={"system_prompt": "A new system prompt."},
                                     at_round=1)]
    run = exp.run(seed=1, max_rounds=2, out=tmp_path)
    turns = [e for e in run.events if e["type"] == "turn_ended"]
    assert [t["yield_kind"] for t in turns] == ["end_turn", "end_turn"]
    assert system_text(stub.kwargs[2]) == "A new system prompt." and thinking_sigs(stub.kwargs[2]) == []
    assert_round1_traces_on_disk(run)


def test_opt_in_kept_thinking_still_counts_when_it_is_sent(tmp_path):
    # control: with carry_thinking and nothing invalidating, the 15k tokens are really sent, so
    # the same limit fails the turn: the cleanup above is what made those runs fit
    stub = ThinkingClaude(pad=PAD, pad_on=(2,))
    run = claude_experiment(stub, prefix="research", carry_thinking=True, context_limit_tokens=LIMIT,
                            overflow="fail_turn").run(seed=1, max_rounds=2, out=tmp_path)
    turns = [e for e in run.events if e["type"] == "turn_ended"]
    assert [t["yield_kind"] for t in turns] == ["end_turn", "error"]
    assert "context_limit" in turns[1]["error"]
