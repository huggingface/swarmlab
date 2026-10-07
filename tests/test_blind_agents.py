"""FlagGame blind agents (WP16): no crop, no guess tool, out of scores and belief denominators."""
import asyncio

import pytest

from swarmlab import Board
from swarmlab.blobs import BlobStore
from swarmlab.executor import RoundExecutor
from swarmlab.rng import derive
from swarmlab.world.flaggame import BLIND_NOTE, CROP_HEADER, FlagGame, parse_observation

from .helpers import flag_experiment

AGENTS = ["a000", "a001", "a002", "a003"]


def reset(world, agents=AGENTS):
    world.reset(derive(1, "world"), list(agents))
    return world


def executor(world, tmp_path):
    return RoundExecutor(run="r", round=1, world=world, board=Board(topology="broadcast"),
                         blobs=BlobStore(tmp_path / "blobs"), agents=list(world.agents))


def test_spec_unchanged_when_unset():
    plain = FlagGame().spec()
    assert FlagGame(blind_agents=[]).spec() == plain
    assert FlagGame(blind_agents=0, blind_may_guess=False).spec() == plain
    assert FlagGame(blind_agents=1).spec()["params"]["blind_agents"] == 1
    assert FlagGame(blind_agents=["a002"], blind_may_guess=True).spec()["params"]["blind_may_guess"]


def test_int_and_list_forms():
    assert reset(FlagGame(blind_agents=2)).blind == ["a000", "a001"]
    assert reset(FlagGame(blind_agents=["a003", "a001"])).blind == ["a001", "a003"]
    assert reset(FlagGame()).blind == []


@pytest.mark.parametrize("kw,match", [
    ({"blind_agents": 4}, "no agent with a crop"),
    ({"blind_agents": AGENTS}, "nobody has a crop"),
    ({"blind_agents": ["a009"]}, "not in the game"),
    ({"blind_agents": 1, "crop_overrides": {"a000": [0, 0]}}, "blind agents"),
])
def test_bad_blind_sets(kw, match):
    with pytest.raises(ValueError, match=match):
        reset(FlagGame(**kw))


@pytest.mark.parametrize("bad", [-1, True, "a000", ["a000", "a000"]])
def test_bad_blind_param(bad):
    with pytest.raises(ValueError):
        FlagGame(blind_agents=bad)


def test_text_observation():
    plain, blind = reset(FlagGame()), reset(FlagGame(blind_agents=1))
    obs = blind.observe("a000")
    text = obs.parts[0].text
    assert text.endswith("\n\n" + BLIND_NOTE) and CROP_HEADER not in text and obs.private == {}
    candidates, crop = parse_observation(text)
    assert candidates == plain.candidates and crop == []
    # everything else is what it would be without blind agents
    assert blind.truth == plain.truth and blind.candidates == plain.candidates
    for a in AGENTS[1:]:
        assert blind.observe(a) == plain.observe(a)
    assert "Some agents have no crop" in blind.description()
    assert "no crop" not in plain.description()


def test_image_observation():
    blind = reset(FlagGame(modality="image", blind_agents=["a001"]))
    parts = blind.observe("a001").parts
    images = [p for p in parts if p.type == "image"]
    assert len(images) == blind.n_candidates
    assert parts[0].text.startswith("Candidate flags:\nCandidates A, B")
    assert "crop" not in parts[0].text and parts[-1].text == BLIND_NOTE
    assert not any(p.type == "text" and p.text == CROP_HEADER for p in parts)
    sighted = blind.observe("a000").parts
    assert len([p for p in sighted if p.type == "image"]) == blind.n_candidates + 1
    hinted = reset(FlagGame(modality="image", image_text_hint=True, blind_agents=1))
    text = "\n".join(p.text for p in hinted.observe("a000").parts if p.type == "text")
    assert parse_observation(text) == (hinted.candidates, [])


def test_tools_and_guesses(tmp_path):
    world = reset(FlagGame(blind_agents=1))
    ex = executor(world, tmp_path)
    assert "guess" not in [s.name for s in ex.schemas("a000")]
    assert "guess" in [s.name for s in ex.schemas("a001")]
    res = asyncio.run(ex.call("a000", "guess", {"candidate": "A"}))
    assert not res.ok and res.error == "not_allowed"
    assert world.guess("a000", "A").accepted is False  # the world refuses it too
    assert world.guess("a000", "A").feedback["error"] == "blind agents may not guess"
    may = reset(FlagGame(blind_agents=1, blind_may_guess=True))
    assert "guess" in [s.name for s in executor(may, tmp_path).schemas("a000")]
    assert may.guess("a000", "A").accepted


def test_score_verify_and_belief_hook():
    world = reset(FlagGame(blind_agents=1, blind_may_guess=True))
    world.guess("a000", world.truth)  # a blind agent's guess never counts
    world.guess("a001", world.truth)
    assert world.score() == {"accuracy": pytest.approx(1 / 3), "n_guessed": 1,
                             "truth": world.truth, "n_blind": 1}
    v = world.verify()
    assert set(v["crops"]) == {"a001", "a002", "a003"} and v["blind"] == ["a000"]
    assert world.excluded_from_belief() == ["a000"]
    plain = reset(FlagGame())
    assert "n_blind" not in plain.score() and "blind" not in plain.verify()
    assert plain.excluded_from_belief() == []
    with pytest.raises(ValueError, match="blind"):
        world.patch_private("a000", {"crop": [0, 0]})


def test_run_denominators_and_replay(tmp_path):
    exp = flag_experiment(4, world=FlagGame(blind_agents=1), medium=Board(topology="broadcast"))
    run = exp.run(seed=1, max_rounds=3, out=tmp_path)
    m = run.metrics
    for name in ("belief.accuracy", "belief.consensus", "belief.entropy"):
        assert [n for _, _, n in m[name]] == [3, 3, 3]
    assert m["belief.accuracy"][-1][1] == pytest.approx(run.score["accuracy"])
    assert run.score["n_blind"] == 1
    called = [e for e in run.events if e["type"] == "tool_returned" and e["agent"] == "a000"
              and e["result"].get("error") == "not_allowed"]
    assert called  # the scripted aggregator tried to guess and was refused
    run.replay()


def test_blind_task_description():
    world = reset(FlagGame(blind_agents=1))
    text = world.description_for("a000")
    assert "You have no crop of your own" in text and "do not record guesses" in text
    assert "`guess`" not in text
    assert world.description_for("a001") == world.description()
    may = reset(FlagGame(modality="image", blind_agents=1, blind_may_guess=True))
    assert "`guess` tool" in may.description_for("a000") and "image" in may.description_for("a000")
    plain = reset(FlagGame())
    assert all(plain.description_for(a) == plain.description() for a in AGENTS)


def test_prompts_show_blind_description():
    from swarmlab import Experiment
    from swarmlab.participants import LLMAgent
    from swarmlab.prompts_cmd import render_prompts
    from swarmlab.roles import assign

    parts = [LLMAgent(model="fake:reader") for _ in range(3)]
    assign(parts[0], "manager")
    exp = Experiment(name="m", world=FlagGame(blind_agents=1), participants=parts,
                     medium=Board(topology="star"))
    rows = render_prompts(exp, seed=1)
    assert "You have no crop of your own: you rely" in rows[0]["system"]
    assert "guess" not in rows[0]["tools"] and "guess" in rows[1]["tools"]
    assert rows[0]["user"].rstrip().endswith(BLIND_NOTE)
    assert "You privately see" in rows[1]["system"]
