"""M6 §4: LLMAgent(report_json=True) on the fake provider (docs/INTERFACE-M6.md)."""
import json
import pickle
from collections import Counter

import pytest

from swarmlab import Board, Experiment, Run
from swarmlab.participants import LLMAgent
from swarmlab.participants.llm import JSON_PROTOCOL_TEXT, TRANSCRIPT_HEADER, report_schema
from swarmlab.providers.base import text_of
from swarmlab.providers.fake import FakeProvider, _response
from swarmlab.snapshot import SnapshotStore
from swarmlab.world.flaggame import FlagGame

from .helpers import logical
from .test_recovery import _truncate_mid_round


@pytest.fixture
def requests_seen(monkeypatch):
    seen: list = []
    original = FakeProvider.complete

    async def recording(self, request):
        seen.append(request)
        return await original(self, request)

    monkeypatch.setattr(FakeProvider, "complete", recording)
    return seen


def last_user(request):
    return text_of(next(m for m in reversed(request.messages) if m.role == "user").content)


def bad_then_good(request, rng):
    if last_user(request).startswith("Your reply could not be used"):
        return _response(request, rng, [], 'Sure! {"country": "germany", "reason": "black on top"}')
    return _response(request, rng, [], "I think it is Germany.")


def always_bad(request, rng):
    return _response(request, rng, [], '{"country": "Atlantis", "reason": "?"}')


def native_guess(request, rng):
    return _response(request, rng, [("guess", {"candidate": "France"}), ("end_turn", {})])


def world(**kw):
    return FlagGame(flags="real", candidates="names", canvas=[24, 16], crop=[6, 4], **kw)


def experiment(agent, n=2, script=None, **kw):
    providers = {"fake": FakeProvider(script=script)} if script else None
    return Experiment(name="rj", world=world(), participants=[agent] * n, providers=providers,
                      medium=Board(), metrics=["belief.consensus"], **kw)


def turn_events(run, agent="a000", round=1):
    return [e for e in run.events if e.get("agent") == agent and e["round"] == round]


def test_schema_lines():
    assert report_schema(["country", "reason"]) == (
        'Output JSON exactly: {"country":"<one allowed country>","reason":"<one sentence>"}')
    assert report_schema(["country"]) == 'Output JSON exactly: {"country":"<one allowed country>"}'


def test_json_answer_becomes_guess_and_post(tmp_path, requests_seen):
    run = experiment(LLMAgent("fake:country_reporter", report_json=True)).run(
        seed=1, max_rounds=1, out=tmp_path)
    for agent in ("a000", "a001"):
        evs = turn_events(run, agent)
        calls = [e["tool"] for e in evs if e["type"] == "tool_called"]
        assert calls == ["guess", "post", "end_turn"]
        ended = next(e for e in evs if e["type"] == "turn_ended")
        assert ended["yield_kind"] == "end_turn"
        post = next(e for e in run.events if e["type"] == "post" and e["agent"] == agent)
        report = json.loads(post["text"])
        assert list(report) == ["country", "reason"]
        guess = next(e for e in run.events if e["type"] == "action_committed" and e["agent"] == agent)
        assert guess["accepted"] and guess["action"]["args"]["candidate"] == report["country"]
    swarm = [r for r in requests_seen if r.tools]
    assert len(swarm) == 2  # one model call per turn
    sysmsg = swarm[0].messages[0].content
    assert sysmsg.startswith("You must output only valid JSON. No extra keys, no markdown, and no "
                             "text outside the JSON object.\nYou are one player in a flag "
                             "identification game.")
    assert "Choose exactly one country from the allowed countries listed in the user message." in sysmsg
    assert "Tools:" not in sysmsg
    assert last_user(swarm[0]).endswith(report_schema(["country", "reason"]))


def test_m1_report_fields(tmp_path, requests_seen):
    run = experiment(LLMAgent("fake:country_reporter", report_json=True, report_fields=["country"])
                     ).run(seed=1, max_rounds=1, out=tmp_path)
    post = next(e for e in run.events if e["type"] == "post")
    assert list(json.loads(post["text"])) == ["country"]
    assert last_user(next(r for r in requests_seen if r.tools)).endswith(
        'Output JSON exactly: {"country":"<one allowed country>"}')


def test_retry_once_then_canonical_name(tmp_path):
    run = experiment(LLMAgent("fake:x", report_json=True), n=1,
                     script="tests.test_report_json:bad_then_good").run(seed=1, max_rounds=1, out=tmp_path)
    evs = turn_events(run)
    assert sum(1 for e in run.events_all if e.type == "inference_attempt") == 2
    post = next(e for e in run.events if e["type"] == "post")
    assert json.loads(post["text"]) == {"country": "Germany", "reason": "black on top"}
    assert any("report_json:retry" in n for e in evs if e["type"] == "turn_ended"
               for n in e["usage"].get("notes", []))


def test_failed_answers_end_the_turn_without_a_post(tmp_path):
    run = experiment(LLMAgent("fake:x", report_json=True), n=1,
                     script="tests.test_report_json:always_bad").run(seed=1, max_rounds=1, out=tmp_path)
    assert not [e for e in run.events if e["type"] == "post"]
    ended = next(e for e in run.events if e["type"] == "turn_ended")
    assert ended["yield_kind"] == "no_tool"
    assert any(n.startswith("report_json:failed") for n in ended["usage"]["notes"])
    assert sum(1 for e in run.events_all if e.type == "inference_attempt") == 2


def test_native_tool_calls_still_work(tmp_path):
    run = experiment(LLMAgent("fake:x", report_json=True), n=1,
                     script="tests.test_report_json:native_guess").run(seed=1, max_rounds=1, out=tmp_path)
    guess = next(e for e in run.events if e["type"] == "action_committed")
    assert guess["accepted"] and guess["action"]["args"]["candidate"] == "France"


def test_json_protocol_path(tmp_path, requests_seen):
    run = experiment(LLMAgent("fake:country_reporter", report_json=True, tool_protocol="json")).run(
        seed=1, max_rounds=1, out=tmp_path)
    assert [e["tool"] for e in turn_events(run) if e["type"] == "tool_called"] == ["guess", "post", "end_turn"]
    sysmsg = next(r for r in requests_seen if r.tools).messages[0].content
    assert JSON_PROTOCOL_TEXT[:40] not in sysmsg


def test_params_and_estimate_cap():
    assert "report_json" not in LLMAgent("fake:x").spec()["params"]
    assert LLMAgent("fake:x", report_json=True).spec()["params"]["report_json"] is True
    assert LLMAgent("fake:x", report_json=True).calls_per_turn_cap == 2
    assert LLMAgent("fake:x", report_json=True, max_calls=1).calls_per_turn_cap == 1
    with pytest.raises(ValueError):
        LLMAgent("fake:x", report_fields=[])
    rj = Experiment(name="e", world=FlagGame(), participants=[LLMAgent("fake:x", report_json=True)] * 2)
    assert rj.estimate(1, 10, calls_per_turn=20)["calls"] == 2 * 10 * 2


def paper_pairwise(n=4, rounds=12, **kw):
    world = FlagGame(flags="real", candidates="names", canvas=[24, 16], crop=[6, 4], modality="image")
    agent = LLMAgent("fake:country_reporter", memory="received", memory_messages=3, report_json=True,
                     temperature=0.2, max_tokens=200)
    return Experiment(name="pairwise", world=world, participants=[agent] * n,
                      medium=Board(topology={"type": "gossip", "params": {"k": 1}}, delivery="push",
                                   push_limit=3, push_consume=True),
                      metrics=["belief.consensus"], probes=[{"type": "belief", "params": {"every": n}}],
                      options={"scheduler": "one_speaker", "max_rounds": rounds, "commit": "immediate"},
                      **kw)


def agent_state(run, round, agent):
    store = SnapshotStore(run.dir)
    return pickle.loads(store.load(store.read(round))[f"participant:{agent}"])


def test_pairwise_run_memory_matches_deliveries_and_resumes(tmp_path, requests_seen):
    exp = paper_pairwise()
    run = exp.run(seed=2, out=tmp_path / "a")
    events = list(run.events)
    deliveries = [e for e in events if e["type"] == "delivery"]
    posts = {e["post_id"]: e["text"] for e in events if e["type"] == "post"}
    assert posts and len(deliveries) == len(posts)
    for agent in ("a000", "a001", "a002", "a003"):
        st = agent_state(run, 12, agent)
        mine = [posts[d["post_id"]] for d in deliveries if d["recipient"] == agent]
        assert [i["content"] for i in st["received"]] == mine[-3:]  # last H, oldest first
    # every probe saw a crop image and the transcript, also for agents that never spoke
    probes = Counter(e["round"] for e in events if e["type"] == "probe")
    assert probes == {4: 4, 8: 4, 12: 4}
    probe_reqs = [r for r in requests_seen if not r.tools and r.messages[-1].role == "user"
                  and "candidate" in str(r.messages[-1].content)]
    assert probe_reqs and all(any(p.type == "image" for p in r.messages[1].content) for r in probe_reqs)
    assert all(TRANSCRIPT_HEADER in json.dumps([p.text for p in r.messages[1].content])
               for r in probe_reqs)
    # resume after a crash reproduces the run
    crash = exp.run(seed=2, out=tmp_path / "b")
    _truncate_mid_round(crash.dir, 7)
    resumed = Run(crash.dir).resume()
    # budget events differ: the truncated log came from a finished run whose later spend the
    # ledger keeps (an artefact of simulating the crash after the fact)
    same = lambda r: [e for e in logical(r) if e["type"] != "budget"]
    assert same(resumed) == same(run) and resumed.metrics == run.metrics
    Run.load(run.dir)
