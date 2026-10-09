"""FlagGame image modality end to end (docs/INTERFACE-M5.md §4, §5 item 4): participants,
probes, exports."""
import base64
import json

import pyarrow.parquet as pq
import pytest

from swarmlab import Board, Experiment, Run
from swarmlab.export import export_run
from swarmlab.participants import EvidenceAggregator, LLMAgent, Silent
from swarmlab.participants.scripted import Enumerator
from swarmlab.probes import BeliefProbe
from swarmlab.providers.fake import FakeProvider
from swarmlab.world.flaggame import reads_text_grids
from swarmlab.world.render import png_size
from swarmlab.worlds import FlagGame

from .helpers import TEST_PRICING, llm_agent_experiment, logical


@pytest.fixture
def requests_seen(monkeypatch):
    seen: list = []
    original = FakeProvider.complete

    async def recording(self, request):
        seen.append(request)
        return await original(self, request)

    monkeypatch.setattr(FakeProvider, "complete", recording)
    return seen


def image_exp(n=4, hint=True, **kw):
    return llm_agent_experiment(n, world=FlagGame(modality="image", image_text_hint=hint), **kw)


def strip(evs):
    return [{k: v for k, v in e.items() if k != "run"} for e in evs]


# ---- refusals at bind ------------------------------------------------------------------------
@pytest.mark.parametrize("participant", [Silent(), EvidenceAggregator(), Enumerator(),
                                         LLMAgent(model="fake:reader"),
                                         LLMAgent(model="fake:flaggame_reader")])
def test_text_readers_need_the_hint(tmp_path, participant):
    exp = Experiment(name="img", world=FlagGame(modality="image"), participants=[participant] * 2,
                     medium=Board(topology="broadcast"),
                     providers={"fake": FakeProvider(pricing=TEST_PRICING)})
    with pytest.raises(ValueError, match="image_text_hint=True"):
        exp.run(seed=1, max_rounds=1, out=tmp_path)
    # the same participants are fine with the hint, and in text mode
    for world in (FlagGame(modality="image", image_text_hint=True), FlagGame()):
        ok = Experiment(name="img", world=world, participants=[participant] * 2,
                        medium=Board(topology="broadcast"),
                        providers={"fake": FakeProvider(pricing=TEST_PRICING)})
        assert ok.run(seed=1, max_rounds=1, out=tmp_path / str(id(world))).status == "ended"


def test_vision_models_and_other_fake_scripts_pass_the_check():
    exp = image_exp(2, hint=False)
    assert not reads_text_grids(LLMAgent(model="anthropic:claude-haiku-4-5"), exp)
    assert not reads_text_grids(LLMAgent(model="fake:tests.helpers:script_end_turn_first"), exp)
    assert reads_text_grids(LLMAgent(model="fake:reader"), exp)
    assert reads_text_grids(LLMAgent(model="fake:reader"))  # no experiment: default fake
    w = FlagGame(modality="image")
    w.check_participants({"a000": LLMAgent(model="anthropic:claude-haiku-4-5")}, exp)


# ---- acceptance 4: fake:reader with the hint -------------------------------------------------
def test_fake_reader_plays_image_mode_deterministically(tmp_path, requests_seen):
    a = image_exp(4).run(seed=3, max_rounds=3, out=tmp_path / "a")
    b = image_exp(4).run(seed=3, max_rounds=3, out=tmp_path / "b")
    assert a.status == "ended" and a.end_reason == "max_rounds"
    assert strip(logical(a)) == strip(logical(b)) and a.score == b.score
    guesses = [e for e in a.events if e["type"] == "action_committed" and e["action"]["name"] == "guess"]
    assert guesses and all(g["accepted"] for g in guesses)
    # the reader combines the posted crops as in text mode: everyone ends on a candidate
    # containing every crop, which is the truth whenever the crops rule out the rival
    text = llm_agent_experiment(4).run(seed=3, max_rounds=3, out=tmp_path / "t")
    assert a.score["truth"] == text.score["truth"]
    assert a.score["accuracy"] == text.score["accuracy"]
    # the requests carry the images, in the documented order, after "Round 1."
    first = requests_seen[0].messages[1].content
    kinds = [p.type for p in first]
    assert kinds[:2] == ["text", "text"] and kinds.count("image") == 9
    assert first[1].text.startswith("Candidate flags:\nCandidates A, B, C")
    assert base64.b64decode(first[2].image_png_b64).startswith(b"\x89PNG")


def test_image_mode_replay_and_fork(tmp_path, requests_seen):
    run = image_exp(4).run(seed=5, max_rounds=4, out=tmp_path)
    before = len(requests_seen)
    replayed = run.replay()  # raises ReplayMismatch on any difference
    assert replayed.get("score", run.score) == run.score
    assert len(requests_seen) == before  # replay calls no provider
    child = run.fork(at_round=2).run(out=tmp_path / "forks")
    parent = [e for e in logical(run) if e["round"] > 2 and e["type"] != "run_started"]
    kid = [e for e in logical(child) if e["round"] > 2 and e["type"] != "run_started"]
    assert strip(kid) == strip(parent)
    assert Run.load(child.dir).score == run.score


def test_window_memory_drops_images_with_their_rounds(tmp_path, requests_seen):
    exp = image_exp(2, agent_kw={"memory": "window", "window_rounds": 2})
    exp.run(seed=1, max_rounds=4, out=tmp_path)
    last = requests_seen[-1]
    users = [m for m in last.messages if m.role == "user"]
    assert [m.content[0].text for m in users] == ["Round 3.", "Round 4."]
    images = sum(1 for m in last.messages if not isinstance(m.content, str)
                 for p in m.content if p.type == "image")
    assert images == 2 * 9


# ---- probes ------------------------------------------------------------------------------------
@pytest.mark.parametrize("hint", [True, False])
def test_probe_context_carries_images_and_names(hint):
    from swarmlab.view import View
    w = FlagGame(modality="image", image_text_hint=hint)
    from swarmlab.rng import derive
    w.reset(derive(1, "world"), ["a000"])
    view = View(round=1, agent="a000", observation=w.observe("a000"), outcomes=[], pushed=[],
                tools=[], description=w.description())
    agent = LLMAgent(model="fake:reader")
    msg = agent.round_message(view)
    assert [p for p in msg.content if p.type == "image"] == [p for p in view.observation.parts
                                                             if p.type == "image"]
    assert BeliefProbe().candidates_from_context([msg]) == list(w.candidates)


def test_probes_run_in_image_mode(tmp_path):
    run = image_exp(3, probes=[BeliefProbe()]).run(seed=2, max_rounds=2, out=tmp_path)
    probes = run.probes["belief"]
    assert probes and all(ok for *_, ok in probes)


# ---- exports -----------------------------------------------------------------------------------
def test_export_sessions_label_images_and_inference_keeps_the_hash(tmp_path):
    run = image_exp(2).run(seed=1, max_rounds=2, out=tmp_path)
    out = export_run(run.dir)
    lines = [json.loads(x) for x in (out / "sessions" / "a000.jsonl").read_text().splitlines()]
    users = [e["message"] for e in lines if e.get("type") == "message"
             and e["message"]["role"] == "user"]
    blocks = [b for u in users if isinstance(u["content"], list) for b in u["content"]]
    assert not any(b["type"] == "image" for b in blocks)
    labels = [b["text"] for b in blocks if b["text"].startswith("[image:")]
    assert labels[:8] == ["[image: PNG 144×96]"] * 8 and labels[8] == "[image: PNG 48×36]"
    assert "iVBOR" not in (out / "sessions" / "a000.jsonl").read_text()
    inf = pq.read_table(out / "tables" / "inference.parquet").to_pylist()
    swarm = [r for r in inf if r["category"] == "swarm"]
    assert swarm and all(r["request"] == f"sha256:{r['request_hash']}" for r in swarm)
    for r in swarm:
        blob = out / "raw" / "blobs" / r["request_hash"][:2] / r["request_hash"]
        assert blob.exists()
    # text-mode exports still inline requests
    text = llm_agent_experiment(2).run(seed=1, max_rounds=1, out=tmp_path / "t")
    tinf = pq.read_table(export_run(text.dir) / "tables" / "inference.parquet").to_pylist()
    assert all(r["request"].startswith("{") for r in tinf if r["request"])


def test_png_sizes_follow_cell_px():
    w = FlagGame(modality="image", cell_px=4)
    from swarmlab.rng import derive
    w.reset(derive(1, "world"), ["a000"])
    imgs = [p for p in w.observe("a000").parts if p.type == "image"]
    assert png_size(base64.b64decode(imgs[0].image_png_b64)) == (48, 32)
    assert png_size(base64.b64decode(imgs[-1].image_png_b64)) == (16, 12)
