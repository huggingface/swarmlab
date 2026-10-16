"""Anthropic prompt caching: what each setting sends, cache-write usage and cost, and that the
setting changes neither request hashes nor existing spec hashes (stub client; no paid calls)."""
import asyncio
import json
from types import SimpleNamespace

import pyarrow.parquet as pq
import pytest

from swarmlab import Board, Experiment
from swarmlab.budget import Gate, HardCeilingReached, Ledger
from swarmlab.participants import LLMAgent
from swarmlab.providers.anthropic import AnthropicProvider, build_kwargs, parse_message
from swarmlab.providers.base import ChatMessage, Usage, usage_cost
from swarmlab.providers.fake import FakeProvider
from swarmlab.providers.openai_compat import OpenAICompatProvider
from swarmlab.registry import build
from swarmlab.spec import Budget, spec_hash
from swarmlab.world.flaggame import FlagGame

from .test_providers import GUESS, StubMessages, anthropic_message, req
from .test_reasoning import ThinkingClaude

M5 = {"type": "ephemeral", "ttl": "5m"}
H1 = {"type": "ephemeral", "ttl": "1h"}


def sent(provider_kw, request=None):
    """The `messages.create` kwargs an `AnthropicProvider(**provider_kw)` sends for `request`."""
    p = AnthropicProvider(**provider_kw)
    p._client = SimpleNamespace(messages=StubMessages([anthropic_message()]))
    asyncio.run(p.complete(request or req(tools=[GUESS])))
    return p._client.messages.kwargs[0]


# ---- what is sent ---------------------------------------------------------------------------------
def test_default_is_prefix_with_5m_on_system_and_top_level():
    kw = sent({})
    assert kw["system"] == [{"type": "text", "text": "You are a000.", "cache_control": M5}]
    assert kw["cache_control"] == M5
    assert sent({"prompt_cache": True}) == kw
    assert sent({"prompt_cache": "prefix", "cache_ttl": "5m"}) == kw
    # nothing else moves: the messages and tools are the plain mapping's
    plain = build_kwargs(req(tools=[GUESS]))
    assert {k: v for k, v in kw.items() if k not in ("system", "cache_control")} == {
        k: v for k, v in plain.items() if k != "system"}


def test_one_hour_ttl_goes_on_both_markers():
    kw = sent({"cache_ttl": "1h"})
    assert kw["system"][-1]["cache_control"] == H1 and kw["cache_control"] == H1
    kw = sent({"prompt_cache": "auto", "cache_ttl": "1h"})
    assert kw["system"] == "You are a000." and kw["cache_control"] == H1


def test_auto_sends_only_the_top_level_breakpoint():
    kw = sent({"prompt_cache": "auto"})
    assert kw["system"] == "You are a000." and kw["cache_control"] == M5


@pytest.mark.parametrize("off", [False, "none"])
def test_off_sends_exactly_the_plain_mapping(off):
    kw = sent({"prompt_cache": off})
    assert "cache_control" not in kw and kw["system"] == "You are a000."
    assert kw == build_kwargs(req(tools=[GUESS]))


def test_prefix_without_system_text_sends_only_the_top_level_one():
    r = req(messages=[ChatMessage(role="user", content="Round 1.")])
    kw = sent({}, r)
    assert "system" not in kw and kw["cache_control"] == M5


def test_several_system_messages_are_one_marked_block():
    r = req(messages=[ChatMessage(role="system", content="A."), ChatMessage(role="system", content="B."),
                      ChatMessage(role="user", content="Round 1.")])
    assert build_kwargs(r, prompt_cache="prefix")["system"] == [
        {"type": "text", "text": "A.\n\nB.", "cache_control": M5}]


def test_extra_still_wins_and_both_markers_carry_its_ttl():
    # a top-level override goes on the system block too: never 5m before 1h (the API's order rule)
    kw = build_kwargs(req(extra={"cache_control": H1}), prompt_cache="prefix")
    assert kw["cache_control"] == H1 and kw["system"][-1]["cache_control"] == H1
    kw = sent({"cache_ttl": "1h"}, req(extra={"cache_control": M5}))
    assert kw["cache_control"] == M5 and kw["system"][-1]["cache_control"] == M5
    # a marker without a ttl is the API's 5m default, copied as given
    kw = build_kwargs(req(extra={"cache_control": {"type": "ephemeral"}}), prompt_cache="prefix")
    assert kw["system"][-1]["cache_control"] == {"type": "ephemeral"}
    # under "none" the extra's marker is sent as is and nothing is added
    kw = build_kwargs(req(extra={"cache_control": H1}), prompt_cache="none")
    assert kw["cache_control"] == H1 and kw["system"] == "You are a000."
    # an extra `system` still replaces the marked block
    kw = build_kwargs(req(extra={"system": "S."}), prompt_cache="prefix")
    assert kw["system"] == "S." and kw["cache_control"] == M5


def test_bad_settings_are_rejected():
    with pytest.raises(ValueError, match="prompt_cache"):
        AnthropicProvider(prompt_cache="always")
    with pytest.raises(ValueError, match="cache_ttl"):
        AnthropicProvider(cache_ttl="10m")
    with pytest.raises(ValueError, match="prompt_cache"):
        build_kwargs(req(), prompt_cache="on")


# ---- usage and cost --------------------------------------------------------------------------------
def test_usage_parses_reads_and_writes():
    msg = anthropic_message()
    msg.usage = SimpleNamespace(input_tokens=100, output_tokens=20, cache_read_input_tokens=50,
                                cache_creation_input_tokens=200)
    usage = parse_message(msg)[2]
    assert usage == Usage(prompt_tokens=350, completion_tokens=20, cached_prompt_tokens=50,
                          cache_write_tokens=200)


async def test_complete_prices_writes_at_the_ttl_factor():
    for ttl, factor in (("5m", 1.25), ("1h", 2.0)):
        msg = anthropic_message()
        msg.usage = SimpleNamespace(input_tokens=100, output_tokens=20, cache_read_input_tokens=50,
                                    cache_creation_input_tokens=200)
        p = AnthropicProvider(cache_ttl=ttl)
        p._client = SimpleNamespace(messages=StubMessages([msg]))
        resp = await p.complete(req())
        assert p.write_factor == factor
        assert resp.cost_usd == pytest.approx(
            (100 * 1.0 + 50 * 0.10 + 200 * 1.0 * factor + 20 * 5.0) / 1e6)


def test_cost_formula_and_base_factor():
    u = Usage(prompt_tokens=1000, cached_prompt_tokens=400, cache_write_tokens=300,
              completion_tokens=200)
    row = (1.0, 5.0, 0.1)
    assert usage_cost(u, row, 1.25) == pytest.approx((300 + 400 * 0.1 + 300 * 1.25 + 1000) / 1e6)
    # the base class (OpenAI-compatible, fake) has no write premium
    for p in (OpenAICompatProvider("hf", pricing={"m": row}), FakeProvider()):
        assert p.write_factor == 1.0
        assert p.cost(req(model=f"{p.name}:m"), u) == pytest.approx(usage_cost(u, row, 1.0))
    # writes never count beyond the prompt (reads first, writes clamped to the rest)
    over = Usage(prompt_tokens=100, cached_prompt_tokens=80, cache_write_tokens=50)
    assert usage_cost(over, row, 2.0) == pytest.approx((80 * 0.1 + 20 * 2.0) / 1e6)
    # an explicit factor wins over the provider attribute
    assert FakeProvider().cost(req(model="fake:m"), u, 2.0) == pytest.approx(
        usage_cost(u, FakeProvider().model_pricing("m"), 2.0))


def one_hour_writes(w5=0, w1=0, **usage):
    """A response whose usage reports `w5` 5-minute and `w1` 1-hour cache-write tokens."""
    msg = anthropic_message()
    fields = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, **usage}
    msg.usage = SimpleNamespace(**fields, cache_creation_input_tokens=w5 + w1,
                                cache_creation=SimpleNamespace(ephemeral_5m_input_tokens=w5,
                                                               ephemeral_1h_input_tokens=w1))
    return msg


async def test_one_hour_override_in_auto_mode_is_billed_at_two():
    r = req(extra={"cache_control": H1})
    p = AnthropicProvider(prompt_cache="auto")  # constructed with 5m
    p._client = SimpleNamespace(messages=StubMessages([one_hour_writes(w1=1000)]))
    resp = await p.complete(r)
    assert p._client.messages.kwargs[0]["cache_control"] == H1
    assert resp.cost_usd == pytest.approx(1000 * 2.0 / 1e6)  # not 1.25x
    # without the per-TTL split the fallback is the TTL the request sent, not the constructor's
    msg = one_hour_writes(w1=1000)
    del msg.usage.cache_creation
    p._client = SimpleNamespace(messages=StubMessages([msg]))
    assert (await p.complete(r)).cost_usd == pytest.approx(1000 * 2.0 / 1e6)
    # ... and with no marker sent at all, the constructor's TTL
    q = AnthropicProvider(prompt_cache="none", cache_ttl="1h")
    q._client = SimpleNamespace(messages=StubMessages([msg]))
    assert (await q.complete(req())).cost_usd == pytest.approx(1000 * 2.0 / 1e6)


async def test_mixed_ttl_writes_are_billed_exactly():
    msg = one_hour_writes(w5=600, w1=400)
    msg.usage.input_tokens, msg.usage.output_tokens, msg.usage.cache_read_input_tokens = 100, 20, 50
    p = AnthropicProvider()
    p._client = SimpleNamespace(messages=StubMessages([msg]))
    resp = await p.complete(req())
    assert resp.usage.cache_write_tokens == 1000 and resp.usage.prompt_tokens == 1150
    assert resp.cost_usd == pytest.approx(
        (100 * 1.0 + 50 * 0.10 + 600 * 1.0 * 1.25 + 400 * 1.0 * 2.0 + 20 * 5.0) / 1e6)


# ---- the gate's reservation -----------------------------------------------------------------------
def big(**kw):
    """10,000 estimated prompt tokens, max_tokens=1."""
    return req(messages=[ChatMessage(role="user", content="x" * 40_000)], max_tokens=1, **kw)


def test_max_cost_reserves_the_write_premium_of_the_ttl_sent():
    r = big()
    assert AnthropicProvider().estimate_prompt_tokens(r) == 10_000
    at = {f: (10_000 * 1.0 * f + 1 * 5.0) / 1e6 for f in (1.0, 1.25, 2.0)}
    assert AnthropicProvider().max_cost(r) == pytest.approx(at[1.25])
    assert AnthropicProvider(cache_ttl="1h").max_cost(r) == pytest.approx(at[2.0])
    assert AnthropicProvider(prompt_cache="auto", cache_ttl="1h").max_cost(r) == pytest.approx(at[2.0])
    # nothing asks for caching: 1x as before, whatever the TTL
    assert AnthropicProvider(prompt_cache="none", cache_ttl="1h").max_cost(r) == pytest.approx(at[1.0])
    # an extra that sends a 1h marker reserves 2x, wherever the marker is
    assert AnthropicProvider(prompt_cache="auto").max_cost(big(extra={"cache_control": H1})) == \
        pytest.approx(at[2.0])
    assert AnthropicProvider(prompt_cache="none").max_cost(big(extra={"cache_control": H1})) == \
        pytest.approx(at[2.0])
    system = {"system": [{"type": "text", "text": "S.", "cache_control": H1}]}
    assert AnthropicProvider(prompt_cache="none").max_cost(big(extra=system)) == pytest.approx(at[2.0])
    # a 5m override of a 1h provider leaves only 5m markers
    assert AnthropicProvider(cache_ttl="1h").max_cost(big(extra={"cache_control": M5})) == \
        pytest.approx(at[1.25])
    # base providers: no write premium
    assert FakeProvider().max_cost(big(model="fake:m")) == pytest.approx(at[1.0])


def priced(**kw):
    p = AnthropicProvider(pricing={"claude-haiku-4-5": (1.0, 5.0, 0.1)}, **kw)
    p._client = SimpleNamespace(messages=StubMessages([one_hour_writes(w1=10_000, output_tokens=1)]))
    return p


async def test_gate_refuses_a_call_whose_writes_could_exceed_the_ceiling():
    p = priced(cache_ttl="1h")
    g = Gate({"anthropic": p}, Ledger(), Budget(hard_usd=0.015))
    with pytest.raises(HardCeilingReached):
        async with g.admit(big(), "swarm"):
            pytest.fail("must not be admitted")
    assert p._client.messages.kwargs == [] and p.calls == 0  # the provider was never called
    assert g.ledger.reserved == 0
    # with room for the worst case it is admitted, and the charge stays within the reservation
    g = Gate({"anthropic": p}, Ledger(), Budget(hard_usd=0.021))
    async with g.admit(big(), "swarm") as res:
        g.dispatch(res)
        resp = await p.complete(big())
        res.charge(resp.cost_usd)
    assert res.amount == pytest.approx(0.020005)
    assert resp.cost_usd == pytest.approx(0.020005) and resp.cost_usd <= res.amount
    assert g.ledger.spent["swarm"] == pytest.approx(0.020005)


async def test_gate_keeps_one_x_with_caching_off():
    p = priced(prompt_cache="none", cache_ttl="1h")
    g = Gate({"anthropic": p}, Ledger(), Budget(hard_usd=0.015))
    async with g.admit(big(), "swarm") as res:
        assert res.amount == pytest.approx(0.010005)


def test_usage_add_and_serialisation():
    a = Usage(prompt_tokens=10, cached_prompt_tokens=2, cache_write_tokens=3)
    b = Usage(prompt_tokens=5, completion_tokens=1, cache_write_tokens=4)
    assert a + b == Usage(prompt_tokens=15, completion_tokens=1, cached_prompt_tokens=2,
                          cache_write_tokens=7)
    # 0 is left out, so usage dicts and response blobs without writes are as before
    assert Usage(prompt_tokens=1).model_dump() == {
        "prompt_tokens": 1, "completion_tokens": 0, "cached_prompt_tokens": 0, "reasoning_tokens": 0}
    assert a.model_dump()["cache_write_tokens"] == 3
    assert Usage.model_validate_json(a.model_dump_json()) == a
    assert Usage(**Usage().model_dump()) == Usage()


# ---- hashes ---------------------------------------------------------------------------------------
def claude_experiment(stub, **provider_kw):
    p = AnthropicProvider(**provider_kw)
    p._client = SimpleNamespace(messages=stub)
    return Experiment(name="cache", world=FlagGame(),
                      participants=[LLMAgent(model="anthropic:claude-haiku-4-5")],
                      medium=Board(topology="broadcast"), providers={"anthropic": p})


def test_request_hashes_are_the_same_with_caching_on_and_off(tmp_path):
    hashes, sent_kw = {}, {}
    for setting in ("prefix", "auto", "none"):
        stub = ThinkingClaude()
        run = claude_experiment(stub, prompt_cache=setting).run(seed=1, max_rounds=2,
                                                                out=tmp_path / setting)
        hashes[setting] = [e.request_hash for e in run.events_all if e.type == "inference_attempt"]
        sent_kw[setting] = stub.kwargs
    assert len(hashes["none"]) == 4
    assert hashes["prefix"] == hashes["auto"] == hashes["none"]
    assert all("cache_control" in kw for kw in sent_kw["prefix"])  # ... while the bodies differ
    assert not any("cache_control" in kw for kw in sent_kw["none"])


MAIN_PARAMS = {"pricing": None, "concurrency": 8, "timeout_s": 90.0, "max_retries": 2}


def test_defaults_stay_out_of_the_provider_spec():
    assert AnthropicProvider().spec() == {"type": "anthropic", "params": MAIN_PARAMS}
    assert AnthropicProvider(prompt_cache="prefix", cache_ttl="5m").spec()["params"] == MAIN_PARAMS
    assert AnthropicProvider(prompt_cache=True).spec()["params"] == MAIN_PARAMS
    spec = AnthropicProvider(prompt_cache="auto", cache_ttl="1h").spec()
    assert spec["params"] == {**MAIN_PARAMS, "prompt_cache": "auto", "cache_ttl": "1h"}
    q = build(spec, "swarmlab.providers")
    assert (q.prompt_cache, q.cache_ttl, q.write_factor) == ("auto", "1h", 2.0)
    assert AnthropicProvider(prompt_cache=False).spec()["params"]["prompt_cache"] is False
    assert build(AnthropicProvider(prompt_cache=False).spec(), "swarmlab.providers").prompt_cache == "none"


YAML = """\
name: cache
budget: {{soft_usd: 0, hard_usd: 1.0, measurement_usd: 0}}
{providers}
arms:
  A:
    world: {{type: flaggame, params: {{n_candidates: 4}}}}
    participants:
      - {{type: llm, count: 2, params: {{model: "anthropic:claude-haiku-4-5"}}}}
    medium: {{topology: broadcast}}
"""


def yaml_experiment(tmp_path, name, providers):
    path = tmp_path / f"{name}.yaml"
    path.write_text(YAML.format(providers=providers))
    return Experiment.from_yaml(path, "A")


def test_spec_round_trip_of_the_provider_params(tmp_path):
    exp = yaml_experiment(tmp_path, "set", "providers:\n  anthropic: {type: anthropic, params: "
                                            "{prompt_cache: auto, cache_ttl: 1h}}")
    p = exp.providers["anthropic"]
    assert (p.prompt_cache, p.cache_ttl, p.write_factor) == ("auto", "1h", 2.0)
    spec = exp.to_spec(seed=0, max_rounds=1)
    params = spec.providers["anthropic"].params
    assert params["prompt_cache"] == "auto" and params["cache_ttl"] == "1h"
    again = Experiment.from_spec(spec)
    assert spec_hash(again.to_spec(seed=0, max_rounds=1)) == spec_hash(spec)
    assert again.providers["anthropic"].prompt_cache == "auto"


def test_spec_without_the_setting_hashes_as_before(tmp_path):
    bare = yaml_experiment(tmp_path, "bare", "providers:\n  anthropic: {type: anthropic, params: {}}")
    explicit = yaml_experiment(tmp_path, "explicit", "providers:\n  anthropic: {type: anthropic, "
                                                     "params: {prompt_cache: prefix, cache_ttl: 5m}}")
    off = yaml_experiment(tmp_path, "off", "providers:\n  anthropic: {type: anthropic, "
                                           "params: {prompt_cache: none}}")
    h = [spec_hash(e.to_spec(seed=0, max_rounds=1)) for e in (bare, explicit, off)]
    assert h[0] == h[1] != h[2]
    # what enters the hash is exactly the provider spec from before the setting existed
    assert bare.to_spec(seed=0, max_rounds=1).providers["anthropic"].params == MAIN_PARAMS
    assert bare.providers["anthropic"].prompt_cache == "prefix"
    # and a spec with no `providers:` at all has none to hash
    none = yaml_experiment(tmp_path, "none", "")
    assert none.to_spec(seed=0, max_rounds=1).providers == {}


# ---- export ---------------------------------------------------------------------------------------
class WritingClaude(ThinkingClaude):
    """`ThinkingClaude` whose first call writes 300 prompt tokens to the cache, later ones read."""

    async def create(self, **kw):
        msg = await super().create(**kw)
        first = len(self.kwargs) == 1
        msg.usage.cache_creation_input_tokens = 300 if first else 0
        msg.usage.cache_read_input_tokens = 0 if first else 300
        return msg


def test_export_has_cache_writes(tmp_path):
    run = claude_experiment(WritingClaude()).run(seed=1, max_rounds=2, out=tmp_path / "run")
    responses = [e for e in run.events_all if e.type == "inference_response"]
    assert responses[0].usage["cache_write_tokens"] == 300
    assert "cache_write_tokens" not in responses[1].usage  # 0 is left out of the event
    out = run.export(tmp_path / "export")
    rows = pq.read_table(out / "tables" / "inference.parquet").to_pylist()
    assert [r["cache_write_tokens"] for r in rows] == [300, 0, 0, 0]
    assert [r["prompt_tokens"] for r in rows] == [400] * 4  # reads and writes are in it
    assert rows[0]["cost_usd"] == pytest.approx((100 + 300 * 1.25 + 60 * 5.0) / 1e6)
    turns = pq.read_table(out / "tables" / "turns.parquet").to_pylist()
    assert [t["cache_write_tokens"] for t in turns] == [300, 0]
    assert pq.read_table(out / "tables" / "turns.parquet").schema.names[-1] == "cache_write_tokens"
    session_file = min((out / "sessions").glob("*.jsonl"))
    session = [json.loads(line) for line in session_file.read_text().splitlines()]
    usages = [m["message"]["usage"] for m in session
              if m.get("type") == "message" and m["message"].get("role") == "assistant"]
    assert usages[0]["cacheWrite"] == 300 and usages[0]["input"] == 100
    assert usages[1]["cacheRead"] == 300 and usages[1]["cacheWrite"] == 0
