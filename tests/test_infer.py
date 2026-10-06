"""Inference through the harness: AgentTools.infer, operational events, cache, usage merge."""
import json

import pytest

from swarmlab import Board, Experiment, Run
from swarmlab.events import OPERATIONAL_TYPES, logical_view
from swarmlab.providers.base import request_hash
from swarmlab.providers.fake import FakeProvider
from swarmlab.world.flaggame import FlagGame

from .helpers import TEST_PRICING, TwiceInferrer, llm_experiment, logical


def events(run):
    return list(run.events_all)


def test_attempt_and_response_events_are_operational(tmp_path):
    run = llm_experiment(4).run(seed=1, max_rounds=2, out=tmp_path)
    evs = events(run)
    attempts = [e for e in evs if e.type == "inference_attempt"]
    responses = [e for e in evs if e.type == "inference_response"]
    assert len(attempts) == len(responses) == 4 * 2 * 2  # 4 agents, 2 rounds, 2 calls per turn
    assert {"inference_attempt", "inference_response"} <= OPERATIONAL_TYPES
    assert all(e["type"] not in OPERATIONAL_TYPES for e in logical_view(evs))
    a = attempts[0]
    assert a.provider == "fake" and a.model == "fake:flaggame_reader" and a.category == "swarm"
    assert a.reserved_usd > 0 and a.call_id.startswith("i0001-")
    by_call = {r.call_id: r for r in responses}
    for a in attempts:
        r = by_call[a.call_id]
        assert r.seq > a.seq and not r.cached and r.cost_usd > 0
        # request and response bodies are blobs; the request blob's sha is the request hash
        from swarmlab.blobs import BlobStore
        from swarmlab.providers.base import ChatRequest, ChatResponse

        blobs = BlobStore(run.dir / "blobs")
        req = ChatRequest.model_validate_json(blobs.get(a.request_hash))
        assert request_hash(req) == a.request_hash
        assert ChatResponse.model_validate_json(blobs.get(r.response_hash)).cost_usd == r.cost_usd
        assert (run.dir / "blobs" / "cache" / a.request_hash).exists()
    # attempts are on disk before the turn's buffered events
    first_turn = next(e.seq for e in evs if e.type == "turn_started")
    assert attempts[0].seq < first_turn
    # agents guessed and the spend is in run.json and the budget events
    assert run.score["n_guessed"] == 4
    spend = run.spend
    assert spend["calls"] == 16 and spend["reserved"] == 0
    assert spend["swarm"] == pytest.approx(sum(r.cost_usd for r in responses))
    budget = [e for e in evs if e.type == "budget"]
    assert [b.round for b in budget] == [1, 2] and budget[-1].calls == 16
    index = (run.dir / "blobs" / "cache" / "index.jsonl").read_text().splitlines()
    assert {json.loads(x)["round"] for x in index} == {1, 2}


def test_usage_merged_into_turn_ended(tmp_path):
    run = llm_experiment(2).run(seed=2, max_rounds=1, out=tmp_path)
    evs = events(run)
    for agent in ("a000", "a001"):
        responses = [e for e in evs if e.type == "inference_response" and e.agent == agent]
        ended = next(e for e in evs if e.type == "turn_ended" and e.agent == agent)
        u = ended.usage
        assert u["inference_calls"] == 2
        assert u["prompt_tokens"] == sum(r.usage["prompt_tokens"] for r in responses)
        assert u["completion_tokens"] == sum(r.usage["completion_tokens"] for r in responses)
        assert u["cost_usd"] == pytest.approx(sum(r.cost_usd for r in responses))
        assert u["cost_usd"] != 123.0  # the participant's own figure is not trusted
        assert u["calls"] == ended.calls  # tool calls from the participant are kept
    # a 2-call turn costs about one cent with the test pricing
    assert 0.005 < run.spend["swarm"] / 2 < 0.02


def test_cache_hit_on_identical_request_within_the_run(tmp_path):
    exp = Experiment(name="twice", world=FlagGame(), participants=[TwiceInferrer()] * 2,
                     medium=Board(), providers={"fake": FakeProvider(pricing=TEST_PRICING)})
    provider = exp.providers["fake"]
    run = exp.run(seed=1, max_rounds=2, out=tmp_path)
    evs = events(run)
    responses = [e for e in evs if e.type == "inference_response"]
    attempts = [e for e in evs if e.type == "inference_attempt"]
    # per agent: round 1 miss + hit, round 2 hit + hit (same request every round)
    assert [r.cached for r in responses if r.agent == "a000"] == [False, True, True, True]
    assert all(a.reserved_usd == 0 for a, r in zip(attempts, responses, strict=True) if r.cached)
    assert all(r.cost_usd == 0 for r in responses if r.cached)
    assert provider.calls == 2 and run.spend["calls"] == 2
    misses = [r.cost_usd for r in responses if not r.cached]
    assert run.spend["swarm"] == pytest.approx(sum(misses))
    # turn_ended carries nominal usage: identical for the hit-only round 2 and the round 1 turn
    ended = [e for e in evs if e.type == "turn_ended" and e.agent == "a000"]
    assert ended[0].usage["cost_usd"] == ended[1].usage["cost_usd"] > 0


def test_replay_makes_no_provider_calls(tmp_path, monkeypatch):
    run = llm_experiment(4).run(seed=3, max_rounds=3, out=tmp_path)
    calls = []
    original = FakeProvider.complete

    async def counting(self, request):
        calls.append(request)
        return await original(self, request)

    monkeypatch.setattr(FakeProvider, "complete", counting)
    loaded = Run.load(run.dir)
    assert calls == []
    assert loaded.replay()["provider_calls"] == 0
    assert loaded.metrics == run.metrics


def test_executor_without_inference_raises():
    import asyncio

    from swarmlab.executor import RoundExecutor
    from swarmlab.providers.base import ChatMessage, ChatRequest

    world = FlagGame()
    ex = RoundExecutor(run="r", round=1, world=world, board=Board(), blobs=None, agents=["a000"])
    with pytest.raises(RuntimeError):
        asyncio.run(ex.infer("a000", ChatRequest(model="fake:x", messages=[
            ChatMessage(role="user", content="hi")])))


def test_fork_copies_cache_for_rounds_up_to_the_fork(tmp_path):
    from swarmlab.inference import InferenceCache

    exp = llm_experiment(3)
    provider = exp.providers["fake"]
    parent = exp.run(seed=4, max_rounds=4, out=tmp_path)
    parent_calls = provider.calls
    child = parent.fork(2).run()
    rounds = {r for _, r in InferenceCache(child.dir / "blobs").entries()}
    assert rounds == {1, 2, 3, 4}  # 1-2 copied, 3-4 written by the child's own live rounds
    copied = InferenceCache(child.dir / "blobs").entries()[: 3 * 2 * 2]
    assert {r for _, r in copied} == {1, 2}
    # rounds 3-4 were not in the child's cache, so the provider was called again
    assert provider.calls - parent_calls == 3 * 2 * 2
    assert logical(child, exclude=("seq", "ts", "run"))[-5:] == logical(parent, exclude=(
        "seq", "ts", "run"))[-5:]
    # the child starts with the parent's ledger at the fork round and adds its own calls
    assert child.spend["calls"] == parent.spend["calls"]
    Run.load(child.dir)
