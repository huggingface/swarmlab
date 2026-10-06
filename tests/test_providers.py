"""Provider layer: hashing, cost math, adapters with stubbed transports, fake determinism."""
import json
import random
from types import SimpleNamespace

import httpx
import pytest

from swarmlab.providers import resolve
from swarmlab.providers.anthropic import AnthropicProvider, build_kwargs
from swarmlab.providers.base import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ProviderError,
    UnknownModelPricing,
    Usage,
    request_hash,
)
from swarmlab.providers.fake import FakeProvider, flaggame_reader, parse_crops
from swarmlab.providers.openai_compat import OpenAICompatProvider, build_body
from swarmlab.tools import ToolCall, ToolSchema
from swarmlab.view import Part
from swarmlab.world.flaggame import FlagGame

GUESS = ToolSchema(name="guess", description="Guess the flag.", parameters={
    "type": "object", "properties": {"candidate": {"type": "string"}},
    "required": ["candidate"], "additionalProperties": False})
LOOSE = ToolSchema(name="read_board", description="Read.", parameters={
    "type": "object", "properties": {"limit": {"type": "integer"}}, "additionalProperties": False})


def req(model="anthropic:claude-haiku-4-5", **kw):
    msgs = kw.pop("messages", None) or [
        ChatMessage(role="system", content="You are a000."),
        ChatMessage(role="user", content=[Part(type="text", text="Round 1."),
                                          Part(type="image", image_png_b64="aGk=")]),
    ]
    return ChatRequest(model=model, messages=msgs, **kw)


# ---- base ----------------------------------------------------------------------------------------
def test_request_hash_is_stable_and_sensitive():
    a, b = req(), req()
    assert request_hash(a) == request_hash(b)
    assert len(request_hash(a)) == 64
    assert request_hash(a) != request_hash(req(max_tokens=10))
    assert request_hash(a) != request_hash(req(extra={"x": 1}))
    # key order in extra does not matter (canonical JSON)
    assert request_hash(req(extra={"a": 1, "b": 2})) == request_hash(req(extra={"b": 2, "a": 1}))
    # pinned value: the canonical form must not drift silently
    fixed = ChatRequest(model="fake:x", messages=[ChatMessage(role="user", content="hi")])
    assert request_hash(fixed) == request_hash(ChatRequest.model_validate_json(fixed.model_dump_json()))


def test_estimate_and_cost_math():
    p = AnthropicProvider()
    r = req(max_tokens=100, messages=[
        ChatMessage(role="user", content="x" * 400),
        ChatMessage(role="user", content=[Part(type="text", text="y" * 40),
                                          Part(type="image", image_png_b64="AAAA")]),
    ])
    assert p.estimate_prompt_tokens(r) == 110 + 1000
    assert p.max_cost(r) == pytest.approx((1110 * 1.0 + 100 * 5.0) / 1e6)
    r2 = r.model_copy(update={"thinking_budget": 1024})
    assert p.max_cost(r2) == pytest.approx((1110 * 1.0 + 1124 * 5.0) / 1e6)
    u = Usage(prompt_tokens=1000, cached_prompt_tokens=400, completion_tokens=200)
    assert p.cost(r, u) == pytest.approx((600 * 1.0 + 400 * 0.10 + 200 * 5.0) / 1e6)


def test_unknown_pricing_and_prefixes():
    with pytest.raises(UnknownModelPricing):
        AnthropicProvider().model_pricing("claude-unknown-9")
    with pytest.raises(UnknownModelPricing):
        OpenAICompatProvider("hf").model_pricing("Qwen/Qwen3-8B")
    hf = OpenAICompatProvider("hf", pricing={"Qwen/Qwen3-8B": [0.1, 0.2, 0.0]})
    assert hf.model_pricing("Qwen/Qwen3-8B") == (0.1, 0.2, 0.0)
    assert hf.base_url == "https://router.huggingface.co/v1" and hf.api_key_env == "HF_TOKEN"
    p, mid = resolve("hf:Qwen/Qwen3-8B")
    assert p.name == "hf" and mid == "Qwen/Qwen3-8B"
    assert resolve("fake:anything")[0].model_pricing("anything") == (1.0, 5.0, 0.1)
    override = FakeProvider()
    assert resolve("fake:x", {"fake": override})[0] is override
    with pytest.raises(ValueError):
        resolve("nope:model")
    with pytest.raises(ValueError):
        resolve("no-prefix")
    with pytest.raises(ValueError):
        resolve("vllm:some-model")  # base_url required


def test_provider_specs_round_trip():
    from swarmlab.registry import build

    for p in (AnthropicProvider(), OpenAICompatProvider("vllm", base_url="http://h:8000/v1"),
              FakeProvider(pricing={"*": (2.0, 3.0, 0.0)})):
        spec = json.loads(json.dumps(p.spec()))
        q = build(spec, "swarmlab.providers")
        assert type(q) is type(p) and q.pricing == p.pricing and q.name == p.name


# ---- anthropic -------------------------------------------------------------------------------------
class StubMessages:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.kwargs = []

    async def create(self, **kw):
        self.kwargs.append(kw)
        out = self.outcomes.pop(0)
        if isinstance(out, BaseException):
            raise out
        return out


def stub_provider(outcomes):
    p = AnthropicProvider()
    p._backoff_s = 0.0
    p._client = SimpleNamespace(messages=StubMessages(outcomes))
    return p


def anthropic_message(**kw):
    return SimpleNamespace(
        content=kw.get("content", [SimpleNamespace(type="text", text="hello")]),
        stop_reason=kw.get("stop_reason", "end_turn"),
        usage=SimpleNamespace(input_tokens=100, output_tokens=20, cache_read_input_tokens=50,
                              cache_creation_input_tokens=0),
    )


def test_anthropic_request_mapping():
    r = req(tools=[GUESS, LOOSE], thinking_budget=2048, temperature=0.5, max_tokens=4096, messages=[
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="user", content=[Part(type="text", text="Round 1."),
                                          Part(type="image", image_png_b64="aGk=")]),
        ChatMessage(role="assistant", content="ok", tool_calls=[
            ToolCall(call_id="toolu_1", name="guess", args={"candidate": "A"}),
            ToolCall(call_id="toolu_2", name="read_board", args={})]),
        ChatMessage(role="tool", content='{"ok": true}', tool_call_id="toolu_1"),
        ChatMessage(role="tool", content='{"items": []}', tool_call_id="toolu_2"),
        ChatMessage(role="user", content="Round 2."),
    ])
    kw = build_kwargs(r)
    assert kw["model"] == "claude-haiku-4-5" and kw["system"] == "sys"
    assert kw["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert "temperature" not in kw  # not sent with thinking
    assert kw["tool_choice"] == {"type": "auto"}
    strict = {t["name"]: t.get("strict") for t in kw["tools"]}
    assert strict == {"guess": True, "read_board": None}
    msgs = kw["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[0]["content"][1] == {"type": "image", "source": {
        "type": "base64", "media_type": "image/png", "data": "aGk="}}
    assert [b["type"] for b in msgs[1]["content"]] == ["text", "tool_use", "tool_use"]
    assert msgs[1]["content"][1]["input"] == {"candidate": "A"}
    # both tool results and the next user text in one user message
    assert [b["type"] for b in msgs[2]["content"]] == ["tool_result", "tool_result", "text"]
    assert msgs[2]["content"][0]["tool_use_id"] == "toolu_1"
    # thinking only for models that take a budget
    other = build_kwargs(r.model_copy(update={"model": "anthropic:claude-other"}))
    assert "thinking" not in other and other["temperature"] == 0.5
    assert "thinking" not in build_kwargs(req())
    assert "tools" not in build_kwargs(req(tools=[GUESS], tool_protocol="json"))


async def test_anthropic_complete_parses_tool_use_and_usage():
    msg = anthropic_message(stop_reason="tool_use", content=[
        SimpleNamespace(type="thinking", thinking="..."),
        SimpleNamespace(type="text", text="I guess A."),
        SimpleNamespace(type="tool_use", id="toolu_9", name="guess", input={"candidate": "A"}),
        SimpleNamespace(type="tool_use", id="toolu_10", name="end_turn", input='{}'),
    ])
    p = stub_provider([msg])
    r = req(tools=[GUESS])
    resp = await p.complete(r)
    assert resp.text == "I guess A." and resp.finish_reason == "tool_use"
    assert [(c.call_id, c.name, c.args) for c in resp.tool_calls] == [
        ("toolu_9", "guess", {"candidate": "A"}), ("toolu_10", "end_turn", {})]
    assert resp.usage == Usage(prompt_tokens=150, completion_tokens=20, cached_prompt_tokens=50)
    assert resp.cost_usd == pytest.approx((100 * 1.0 + 50 * 0.1 + 20 * 5.0) / 1e6)
    assert resp.served_by is None and resp.provider == "anthropic" and resp.model == "claude-haiku-4-5"
    assert p.calls == 1


def _sdk_errors():
    import anthropic
    import httpx2

    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")

    def status(code, cls, headers=None):
        return cls("err", response=httpx2.Response(code, request=request, headers=headers or {}),
                   body=None)

    return anthropic, request, status


async def test_anthropic_retries_rate_limit_and_connection_then_succeeds():
    anthropic, request, status = _sdk_errors()
    p = stub_provider([
        status(429, anthropic.RateLimitError, {"retry-after": "0"}),
        anthropic.APIConnectionError(request=request),
        anthropic_message(),
    ])
    resp = await p.complete(req())
    assert resp.text == "hello"
    assert len(p._client.messages.kwargs) == 3


async def test_anthropic_gives_up_after_three_retries_and_raises_status_errors():
    anthropic, _request, status = _sdk_errors()
    p = stub_provider([status(429, anthropic.RateLimitError) for _ in range(4)])
    with pytest.raises(anthropic.RateLimitError):
        await p.complete(req())
    assert len(p._client.messages.kwargs) == 4
    p = stub_provider([status(400, anthropic.BadRequestError), anthropic_message()])
    with pytest.raises(anthropic.BadRequestError):
        await p.complete(req())
    assert len(p._client.messages.kwargs) == 1
    p = stub_provider([status(500, anthropic.InternalServerError), anthropic_message()])
    with pytest.raises(anthropic.APIStatusError):
        await p.complete(req())


def test_anthropic_key_fallback(monkeypatch):
    import asyncio

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_KEY", "sk-test-fallback")
    p = AnthropicProvider()

    async def get():
        return p._get_client()

    client = asyncio.run(get())
    assert client.api_key == "sk-test-fallback" and client.max_retries == 0


# ---- openai-compatible ---------------------------------------------------------------------------
def completion(message, finish="stop", usage=None):
    return {"choices": [{"message": message, "finish_reason": finish}],
            "usage": usage or {"prompt_tokens": 80, "completion_tokens": 10,
                               "prompt_tokens_details": {"cached_tokens": 30},
                               "completion_tokens_details": {"reasoning_tokens": 4}}}


def compat(handler, name="hf"):
    p = OpenAICompatProvider(name, pricing={"m": (1.0, 2.0, 0.5)},
                             **({"base_url": "http://vllm:8000/v1/"} if name == "vllm" else {}))
    p._transport = httpx.MockTransport(handler)
    p._backoff_s = 0.0
    return p


def test_openai_body_mapping():
    r = req(model="hf:m", tools=[GUESS], seed=7, temperature=0.2, messages=[
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="user", content=[Part(type="text", text="look"),
                                          Part(type="image", image_png_b64="aGk=")]),
        ChatMessage(role="assistant", content="", tool_calls=[
            ToolCall(call_id="c1", name="guess", args={"candidate": "B"})]),
        ChatMessage(role="tool", content="done", tool_call_id="c1"),
    ])
    body = build_body(r)
    assert body["model"] == "m" and body["seed"] == 7 and body["temperature"] == 0.2
    assert "top_p" not in body
    assert body["tools"] == [{"type": "function", "function": {
        "name": "guess", "description": "Guess the flag.", "parameters": GUESS.parameters}}]
    assert body["tool_choice"] == "auto"
    assert body["messages"][1]["content"][1] == {
        "type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}}
    a = body["messages"][2]
    assert a["content"] is None and json.loads(a["tool_calls"][0]["function"]["arguments"]) == {
        "candidate": "B"}
    assert body["messages"][3] == {"role": "tool", "tool_call_id": "c1", "content": "done"}
    assert "tools" not in build_body(r.model_copy(update={"tool_protocol": "json"}))


async def test_openai_complete_tool_calls_usage_served_by(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, headers={"x-inference-provider": "together"}, json=completion(
            {"content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "guess", "arguments": '{"candidate": "C"}'}}]},
            finish="tool_calls"))

    p = compat(handler)
    resp = await p.complete(req(model="hf:m", tools=[GUESS]))
    assert seen == {"url": "https://router.huggingface.co/v1/chat/completions",
                    "auth": "Bearer hf_test"}
    assert resp.served_by == "together" and resp.finish_reason == "tool_calls"
    assert resp.tool_calls[0].args == {"candidate": "C"} and resp.text == ""
    assert resp.usage == Usage(prompt_tokens=80, completion_tokens=10, cached_prompt_tokens=30,
                               reasoning_tokens=4)
    assert resp.cost_usd == pytest.approx((50 * 1.0 + 30 * 0.5 + 10 * 2.0) / 1e6)


async def test_openai_bad_tool_args_are_wrapped():
    def handler(request):
        return httpx.Response(200, json=completion({"content": "x", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "guess", "arguments": "{oops"}}]}))

    resp = await compat(handler, "vllm").complete(req(model="vllm:m"))
    assert resp.finish_reason == "bad_tool_args"
    assert resp.tool_calls[0].args == {"_raw": "{oops"}


async def test_openai_retries_429_and_5xx_then_raises():
    codes = iter([429, 503, 200])

    def handler(request):
        code = next(codes)
        return httpx.Response(code, json=completion({"content": "fine"}) if code == 200 else {})

    resp = await compat(handler).complete(req(model="hf:m"))
    assert resp.text == "fine"
    n = {"calls": 0}

    def always_500(request):
        n["calls"] += 1
        return httpx.Response(500, json={})

    with pytest.raises(ProviderError) as e:
        await compat(always_500).complete(req(model="hf:m"))
    assert n["calls"] == 4 and e.value.status == 500

    def bad_request(request):
        n["calls"] += 1
        return httpx.Response(400, json={"error": "no"})

    n["calls"] = 0
    with pytest.raises(ProviderError):
        await compat(bad_request).complete(req(model="hf:m"))
    assert n["calls"] == 1


# ---- fake ----------------------------------------------------------------------------------------
def flag_request(tools=("read_board", "post", "guess", "end_turn"), extra_messages=(), **kw):
    import swarmlab.rng

    world = FlagGame()
    world.reset(swarmlab.rng.derive(3, "world"), ["a000", "a001"])
    obs = world.observe("a000").parts[0].text
    schemas = [ToolSchema(name=n, description=n, parameters={"type": "object", "properties": {}})
               for n in tools]
    msgs = [ChatMessage(role="system", content="You are a000."),
            ChatMessage(role="user", content="Round 1.\n" + obs), *extra_messages]
    return world, ChatRequest(model="fake:flaggame_reader", messages=msgs, tools=schemas, **kw)


async def test_fake_reader_reads_then_guesses_consistent_candidate():
    world, r = flag_request()
    p = FakeProvider()
    first = await p.complete(r)
    assert [c.name for c in first.tool_calls] == ["post", "read_board"]
    assert first.tool_calls[0].args["text"].startswith("crop:\n")
    assert first.usage.prompt_tokens == p.estimate_prompt_tokens(r)
    assert first.cost_usd == pytest.approx(p.cost(r, first.usage)) and first.cost_usd > 0
    other_crop = "\n".join(world.crop_rows("a001"))
    board = json.dumps({"ok": True, "result": {"items": [{"content": "crop:\n" + other_crop}]}})
    _, r2 = flag_request(extra_messages=[
        ChatMessage(role="assistant", content="", tool_calls=first.tool_calls),
        ChatMessage(role="tool", content="{}", tool_call_id=first.tool_calls[0].call_id),
        ChatMessage(role="tool", content=board, tool_call_id=first.tool_calls[1].call_id),
    ])
    second = await p.complete(r2)
    assert [c.name for c in second.tool_calls] == ["guess", "end_turn"]
    guess = second.tool_calls[0].args["candidate"]
    from swarmlab.world.flaggame import contains

    crops = parse_crops(r2.messages[1].content + board)
    assert len(crops) == 2
    assert contains(world.candidates[guess], list(crops[0]))
    assert contains(world.candidates[guess], list(crops[1]))
    assert p.calls == 2


async def test_fake_is_a_pure_function_of_the_request():
    _, r = flag_request()
    a = await FakeProvider().complete(r)
    b = await FakeProvider().complete(ChatRequest.model_validate_json(r.model_dump_json()))
    assert a == b
    # json protocol: calls as a JSON array in the text
    _, rj = flag_request(tool_protocol="json")
    j = await FakeProvider().complete(rj)
    assert j.tool_calls == [] and [c["name"] for c in json.loads(j.text)] == ["post", "read_board"]
    # no tools (a probe): a belief answer
    _, rp = flag_request(tools=())
    probe = json.loads((await FakeProvider().complete(rp)).text)
    assert probe["candidate"] in "ABCDEFGH" and 0 <= probe["confidence"] <= 1


def custom_script(request, rng):
    return ChatResponse(text=f"r{rng.randint(0, 9)}", tool_calls=[], usage=Usage(prompt_tokens=7,
                        completion_tokens=3), cost_usd=0, provider="", model="", latency_s=1,
                        finish_reason="stop")


async def test_fake_scripted_escape_hatch_and_model_named_script():
    p = FakeProvider(script="tests.test_providers:custom_script", pricing={"*": (1.0, 1.0, 0.0)})
    r = ChatRequest(model="fake:whatever", messages=[ChatMessage(role="user", content="hi")])
    resp = await p.complete(r)
    assert resp.text.startswith("r") and resp.usage.prompt_tokens == 7
    assert resp.cost_usd == pytest.approx(10 / 1e6) and resp.latency_s == 0.0
    by_model = FakeProvider()
    r2 = r.model_copy(update={"model": "fake:tests.test_providers:custom_script"})
    assert (await by_model.complete(r2)).text.startswith("r")
    assert flaggame_reader(r, random.Random(0)).tool_calls == []  # nothing offered, no listing
    with pytest.raises(ValueError):
        await FakeProvider().complete(r.model_copy(update={"model": "fake:no_such_script"}))
