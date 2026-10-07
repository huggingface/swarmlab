"""M6 §3: LLMAgent(memory="received"), push_consume, prepare_probe (docs/INTERFACE-M6.md)."""
import asyncio
import json
import pickle
from collections import Counter

import pytest

from swarmlab import Board, Experiment, Run
from swarmlab.medium.board import Board as RawBoard
from swarmlab.participants import LLMAgent
from swarmlab.participants.llm import TRANSCRIPT_HEADER
from swarmlab.providers.base import ChatResponse, Usage
from swarmlab.providers.fake import FakeProvider
from swarmlab.snapshot import SnapshotStore
from swarmlab.tools import ToolResult
from swarmlab.view import Observation, Part, View
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


class StubTools:
    """Answers every model call with a guess of "B" and records the requests."""

    def __init__(self):
        self.requests = []

    async def infer(self, request, *, category="swarm"):
        self.requests.append(request)
        from swarmlab.tools import ToolCall
        return ChatResponse(text="", tool_calls=[ToolCall(call_id=f"c{len(self.requests)}", name="guess",
                                                          args={"candidate": "B"})],
                            usage=Usage(), cost_usd=0.0, provider="stub", model="m", latency_s=0.0,
                            finish_reason="tool_use") if len(self.requests) % 2 else ChatResponse(
            text="done", tool_calls=[], usage=Usage(), cost_usd=0.0, provider="stub", model="m", latency_s=0.0,
            finish_reason="end_turn")

    async def call(self, name, args=None):
        return ToolResult(call_id="x", ok=True, result={})


def view(r, pushed=(), crop="CROP"):
    return View(round=r, agent="a000", observation=Observation(parts=[Part(type="text", text=crop)]),
                outcomes=[{"tool": "guess", "action_id": "x", "accepted": True, "feedback": {}}],
                pushed=list(pushed), tools=[], description="task")


def item(r, n, text):
    return {"delivery_id": f"d{r:04d}-{n:05d}", "post_id": f"p{r:04d}-{n:04d}", "eligible_round": r,
            "content": text}


def test_keeps_the_last_h_items_in_order_and_reshows_the_crop():
    agent = LLMAgent("fake:x", memory="received", memory_messages=3)
    agent.bind("a000", None)
    tools = StubTools()
    asyncio.run(agent.turn(view(1, [item(1, 0, "m1"), item(1, 1, "m2")]), tools))
    asyncio.run(agent.turn(view(2, [item(1, 0, "m1"), item(1, 1, "m2"), item(2, 0, "m3"), item(2, 1, "m4")],
                                crop="CROP2"), tools))
    assert [i["content"] for i in agent.received] == ["m2", "m3", "m4"]
    req = tools.requests[-2]  # the first call of turn 2
    assert [m.role for m in req.messages] == ["system", "user"]  # no history across turns
    texts = [p.text for p in req.messages[1].content]
    assert texts[0] == "CROP2"  # the current observation, re-shown
    assert texts[1] == f"{TRANSCRIPT_HEADER}\n- m2\n- m3\n- m4"
    assert texts[2] == 'Your previous answers (oldest -> newest): ["B"]'
    assert not any("Round" in t or "Outcomes" in t for t in texts)
    assert agent.own_answers == ["B", "B"]
    # the first turn saw its two pushed items; an agent with none sees an empty list
    assert tools.requests[0].messages[1].content[1].text == f"{TRANSCRIPT_HEADER}\n- m1\n- m2"
    fresh = LLMAgent("fake:x", memory="received")
    fresh.bind("a001", None)
    fresh_tools = StubTools()
    asyncio.run(fresh.turn(view(1), fresh_tools))
    assert fresh_tools.requests[0].messages[1].content[1].text == f"{TRANSCRIPT_HEADER} []"
    # probe context: the same construction
    ctx = agent.probe_context()
    assert [m.role for m in ctx] == ["system", "user"]
    assert [p.text for p in ctx[1].content] == [
        "CROP2", f"{TRANSCRIPT_HEADER}\n- m2\n- m3\n- m4",
        'Your previous answers (oldest -> newest): ["B", "B"]']


def test_ingest_dedupes_and_snapshot_roundtrip():
    agent = LLMAgent("fake:x", memory="received", memory_messages=2)
    assert agent.ingest([item(1, 0, "a"), item(1, 1, "b")]) == 2
    assert agent.ingest([item(1, 1, "b"), item(2, 0, "c")]) == 1
    assert [i["content"] for i in agent.received] == ["b", "c"]
    agent.own_answers = ["X"]
    agent.last_observation = [{"type": "text", "text": "crop"}]
    clone = LLMAgent("fake:x", memory="received", memory_messages=2)
    clone.restore(agent.snapshot())
    assert (clone.received, clone.received_upto, clone.own_answers, clone.last_observation) == (
        agent.received, agent.received_upto, agent.own_answers, agent.last_observation)
    assert clone.ingest([item(2, 0, "c")]) == 0


def test_params_only_when_set():
    assert "memory_messages" not in LLMAgent("fake:x").spec()["params"]
    assert LLMAgent("fake:x", memory="received").spec()["params"]["memory_messages"] == 8
    with pytest.raises(ValueError):
        LLMAgent("fake:x", memory="recent")


def test_board_push_consume_marks_read_and_spec_default():
    assert "push_consume" not in RawBoard().spec()["params"]
    assert RawBoard(push_consume=True).spec()["params"]["push_consume"] is True
    b = RawBoard(push_consume=True)
    from swarmlab.blobs import BlobStore  # noqa: F401 - mark_read needs no blobs
    b.commit_mode = "immediate"
    b._inboxes["a001"] = []
    from swarmlab.medium.board import Delivery
    for n in range(3):
        b._inboxes["a001"].append(Delivery(delivery_id=f"d0001-{n:05d}", post_id=f"p0001-{n:04d}",
                                           recipient="a001", eligible_round=1, content_hash="h"))
    b.mark_read("a001", ["d0001-00000", "d0001-00002"], 1)
    assert [d.delivery_id for d in b.pushable("a001", 2, 10)] == ["d0001-00001"]


def reader_pairwise(n=4, rounds=12):
    agent = LLMAgent("fake:reader", memory="received", memory_messages=3)
    return Experiment(name="pairwise", world=FlagGame(), participants=[agent] * n,
                      medium=Board(topology={"type": "gossip", "params": {"k": 1}}, delivery="push",
                                   push_limit=3, push_consume=True),
                      metrics=["belief.consensus"], probes=[{"type": "belief", "params": {"every": n}}],
                      options={"scheduler": "one_speaker", "max_rounds": rounds, "commit": "immediate"})


def agent_state(run, round, agent):
    store = SnapshotStore(run.dir)
    return pickle.loads(store.load(store.read(round))[f"participant:{agent}"])


def test_pairwise_run_memory_matches_deliveries_and_resumes(tmp_path, requests_seen):
    exp = reader_pairwise()
    run = exp.run(seed=2, out=tmp_path / "a")
    events = list(run.events)
    deliveries = [e for e in events if e["type"] == "delivery"]
    posts = {e["post_id"]: e["text"] for e in events if e["type"] == "post"}
    assert posts and len(deliveries) == len(posts)
    for agent in ("a000", "a001", "a002", "a003"):
        st = agent_state(run, 12, agent)
        mine = [posts[d["post_id"]] for d in deliveries if d["recipient"] == agent]
        assert [i["content"] for i in st["received"]] == mine[-3:]  # last H, oldest first
    # every probe saw the crop and the transcript, also for agents that never spoke
    probes = Counter(e["round"] for e in events if e["type"] == "probe")
    assert probes == {4: 4, 8: 4, 12: 4}
    probe_reqs = [r for r in requests_seen if not r.tools]
    assert probe_reqs  # identical probe requests are answered from the run cache
    assert all("Your crop:" in json.dumps([p.text for p in r.messages[1].content]) for r in probe_reqs)
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
