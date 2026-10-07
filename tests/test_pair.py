"""M3a §2: FlagGame crop_overrides, Run.fork(0), Experiment.pair (acceptance 5)."""
import json

import pytest

from swarmlab import Experiment, Run
from swarmlab.experiment import PairedResult
from swarmlab.rng import derive
from swarmlab.spec import RunOptions
from swarmlab.world.flaggame import FlagGame

from .helpers import flag_experiment, llm_agent_experiment, logical

AGENTS = [f"a{i:03d}" for i in range(6)]
EXCLUDE = ("seq", "ts", "run")


def other_crop(seed: int, agent: str = "a003", **world_kw) -> list[int]:
    """A crop position for `agent` whose rows differ from its drawn crop."""
    w = FlagGame(**world_kw)
    w.reset(derive(seed, "world"), AGENTS)
    rows = w.crop_rows(agent)
    for y in range(w.height - w.crop_h + 1):
        for x in range(w.width - w.crop_w + 1):
            truth = w.candidates[w.truth]
            if [r[x:x + w.crop_w] for r in truth[y:y + w.crop_h]] != rows:
                return [y, x]
    raise AssertionError("no other crop")


def body(view):
    """Logical view without run_started (its parent_run/fork_round/run_spec legitimately differ)."""
    return [e for e in view if e["type"] != "run_started"]


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def assert_differs_from_agent_turn(base, patched, agent="a003"):
    """The views agree until `agent`'s first turn_started; round-1 turns of others are equal."""
    i = first_diff(base, patched)
    assert i is not None
    ev = base[i]
    assert ev["type"] == "turn_started" and ev["agent"] == agent and ev["round"] == 1
    turn_types = {"turn_started", "tool_called", "tool_returned", "read", "turn_ended"}

    def others(view):
        return [e for e in view if e["round"] == 1 and e["type"] in turn_types and e["agent"] != agent]

    assert others(base) == others(patched)


# ---- crop_overrides ------------------------------------------------------------------------------
def test_crop_overrides_replace_only_listed_agents():
    plain = FlagGame()
    plain.reset(derive(3, "world"), AGENTS)
    over = FlagGame(crop_overrides={"a003": [0, 0], "a001": [5, 8]})
    over.reset(derive(3, "world"), AGENTS)
    assert over.candidates == plain.candidates and over.truth == plain.truth
    assert over.crops["a003"] == (0, 0) and over.crops["a001"] == (5, 8)
    assert {a: c for a, c in over.crops.items() if a not in ("a001", "a003")} == \
        {a: c for a, c in plain.crops.items() if a not in ("a001", "a003")}
    assert over.verify()["crops"]["a003"] == [0, 0]
    assert over.observe("a003").private == {"crop_y": 0, "crop_x": 0}
    # snapshot carries crops, not the config
    again = FlagGame(crop_overrides={"a003": [0, 0], "a001": [5, 8]})
    again.restore(over.snapshot())
    assert again.crops == over.crops


def test_crop_overrides_validation_and_spec():
    for bad in ({"a000": [6, 0]}, {"a000": [0, 9]}, {"a000": [-1, 0]}, {"a000": [0]},
                {"a000": ["0", 1]}, {"a000": [True, 1]}):
        with pytest.raises(ValueError, match="crop_overrides"):
            FlagGame(crop_overrides=bad)
    FlagGame(crop_overrides={"a000": [5, 8]})  # bottom-right corner fits (8x12, crop 3x4)
    with pytest.raises(ValueError, match="not in the game"):
        FlagGame(crop_overrides={"a099": [0, 0]}).reset(derive(1, "world"), AGENTS)
    assert "crop_overrides" not in FlagGame().spec()["params"]
    spec = FlagGame(crop_overrides={"a002": [1, 2]}).spec()
    assert spec["params"]["crop_overrides"] == {"a002": [1, 2]}
    # round-trips through the experiment spec
    exp = flag_experiment(4, world=FlagGame(crop_overrides={"a002": [1, 2]}))
    back = Experiment.from_spec(exp.to_spec(seed=1, max_rounds=2))
    assert back.world.crop_overrides == {"a002": (1, 2)}
    assert back.to_spec(seed=1, max_rounds=2).world == exp.to_spec(seed=1, max_rounds=2).world


def test_repeat_zero_is_not_in_the_spec():
    assert "repeat" not in RunOptions(seed=1, max_rounds=2).model_dump()
    assert RunOptions(seed=1, max_rounds=2, repeat=2).model_dump()["repeat"] == 2
    exp = flag_experiment(4)
    assert exp.spec_hash(1, 3) == exp.spec_hash(1, 3, repeat=0) != exp.spec_hash(1, 3, repeat=1)


# ---- fork at round 0 -----------------------------------------------------------------------------
@pytest.mark.parametrize("make", [lambda: flag_experiment(6), lambda: llm_agent_experiment(6)])
def test_fork_at_zero_reproduces_parent(tmp_path, make):
    exp = make()
    parent = exp.run(seed=4, max_rounds=4, out=tmp_path)
    child = parent.fork(0).run()
    assert child.id == f"{parent.id}__f0_1"
    first = next(iter(child.events_all))
    assert first.type == "run_started" and first.round == 0
    assert first.parent_run == parent.id and first.fork_round == 0 and first.run == child.id
    assert child.meta["restored"] == {"world": False, "participants": [], "metrics": [], "reset": True}
    assert body(logical(child, EXCLUDE)) == body(logical(parent, EXCLUDE))
    assert child.score == parent.score
    Run.load(child.dir)


def test_fork_at_zero_with_crop_override(tmp_path):
    exp = llm_agent_experiment(6)
    parent = exp.run(seed=4, max_rounds=4, out=tmp_path)
    patched_exp = exp.model_copy(update={"world": FlagGame(crop_overrides={"a003": other_crop(4)})})
    child = parent.fork(0, experiment=patched_exp).run()
    assert child.spec.world.params["crop_overrides"] == {"a003": other_crop(4)}
    assert_differs_from_agent_turn(body(logical(parent, EXCLUDE)), body(logical(child, EXCLUDE)))
    Run.load(child.dir)


def test_fork_at_zero_cannot_change_agent_count(tmp_path):
    parent = flag_experiment(4).run(seed=1, max_rounds=2, out=tmp_path)
    with pytest.raises(ValueError, match="number of participants"):
        parent.fork(0, experiment=flag_experiment(5)).run()


# ---- Experiment.pair -----------------------------------------------------------------------------
def patched_of(exp: Experiment, seed: int) -> Experiment:
    return exp.model_copy(update={"world": FlagGame(crop_overrides={"a003": other_crop(seed)})})


def test_pair_on_fake_llm_agents(tmp_path):
    exp = llm_agent_experiment(6, metrics=["belief.consensus", "belief.accuracy", "comm.read_rate"])
    res = exp.pair(4, 4, patch=patched_of(exp, 4), repeats=2, out=tmp_path / "pair")
    assert isinstance(res, PairedResult)
    assert [len(v) for v in res.runs.values()] == [2, 2, 2]
    assert res.dir == tmp_path / "pair" / "llmagent__s4__pair"
    doc = json.loads((res.dir / "pair.json").read_text())
    assert doc["runs"]["patched"] == ["llmagent__s4__patched_r0", "llmagent__s4__patched_r1"]
    assert res.base[1].id == "llmagent__s4__base_r1" and res.base[1].dir.name == res.base[1].id

    # repeat 0 of the base is exactly a plain run(seed)
    plain = exp.run(4, 4, out=tmp_path / "plain")
    assert logical(res.base[0], EXCLUDE) == logical(plain, EXCLUDE)
    assert res.base[0].meta["spec_hash"] == plain.meta["spec_hash"]
    assert res.base[1].spec.options.repeat == 1

    for i in range(2):
        b = body(logical(res.base[i], EXCLUDE))
        # control pairs: identical logical views on the fake provider
        assert body(logical(res.controls[i], EXCLUDE)) == b
        # patched pairs: identical until the patched agent's first turn
        assert_differs_from_agent_turn(b, body(logical(res.patched[i], EXCLUDE)))
        Run.load(res.patched[i].dir)

    # repeat 1 re-derives sampling: every request carries a seed, so no request equals repeat 0's
    def hashes(run):
        return {e.request_hash for e in run.events_all if e.type == "inference_attempt"}
    assert hashes(res.base[0]).isdisjoint(hashes(res.base[1]))
    assert hashes(res.base[1]) == hashes(res.controls[1])

    eff = res.effect("belief.consensus")
    assert eff["n"] == 2 and eff["n_control"] == 2
    assert eff["control_spread"] == 0.0 and eff["control_diffs"] == [0.0, 0.0]
    vals = [(PairedResult.value(p, "belief.consensus") - PairedResult.value(b, "belief.consensus"))
            for b, p in zip(res.base, res.patched, strict=True)]
    assert eff["diff"] == pytest.approx(sum(vals) / 2)
    assert res.effect("belief.accuracy", round=1)["round"] == 1
    with pytest.raises(KeyError):
        res.effect("belief.consensus", round=99)
    with pytest.raises(FileExistsError):
        exp.pair(4, 4, patch=None, out=tmp_path / "pair")


def test_pair_without_control_and_resume_of_a_repeat(tmp_path):
    exp = flag_experiment(4)
    res = exp.pair(2, 3, patch=None, repeats=2, control=False, out=tmp_path)
    assert res.controls == [] and res.effect("belief.consensus")["control_spread"] is None
    assert res.effect("belief.consensus")["diff"] == 0.0
    # a repeat-1 run resumes and replays with its own streams
    run = res.base[1]
    assert Run(run.dir).resume().status == "ended"
    Run.load(run.dir)
    with pytest.raises(ValueError, match="number of participants"):
        exp.pair(2, 3, patch=flag_experiment(5), out=tmp_path / "x")
    with pytest.raises(ValueError, match="repeats"):
        exp.pair(2, 3, patch=None, repeats=0, out=tmp_path / "y")
