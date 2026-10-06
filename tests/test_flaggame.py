import json
import pickle
import re

import pytest

from swarmlab.ids import ActionId, AgentId
from swarmlab.rng import derive
from swarmlab.world.base import Action
from swarmlab.world.flaggame import (
    FlagGame,
    candidates_containing,
    contains,
    parse_observation,
)

AGENTS = [AgentId(f"a{i:03d}") for i in range(16)]


def make(seed=0, agents=AGENTS, **kw):
    w = FlagGame(**kw)
    w.reset(derive(seed, "world"), agents)
    return w


def commit_guesses(w, guesses, rnd=0):
    acts = [
        (a, ActionId(f"r{rnd}_{a}"), Action(name="guess", args={"candidate": c}))
        for a, c in guesses.items()
    ]
    return w.commit(acts)


def test_reset_deterministic():
    a, b = make(7), make(7)
    assert a.verify() == b.verify()
    assert a.snapshot() == b.snapshot()
    assert make(8).verify() != a.verify()


def test_crops_depend_on_world_rng_and_agents_only():
    a = make(3)
    b = make(3, agents=AGENTS[:4])
    assert a.candidates == b.candidates
    assert {k: a.crops[k] for k in b.crops} == b.crops


@pytest.mark.parametrize("seed", range(20))
def test_rival_fraction_and_distinct(seed):
    w = make(seed)
    truth, rival = w.candidates[w.truth], w.candidates[w.rival]
    diff = sum(t != r for tr, rr in zip(truth, rival) for t, r in zip(tr, rr))
    assert diff == round(0.15 * 8 * 12)
    grids = [tuple(g) for g in w.candidates.values()]
    assert len(set(grids)) == len(grids) == 8
    assert all(len(g) == 8 and all(len(r) == 12 for r in g) for g in grids)
    assert all(set("".join(g)) <= set("rgbykw") for g in grids)


def test_rival_always_differs_even_at_full_similarity():
    w = make(1, rival_similarity=1.0)
    assert w.candidates[w.truth] != w.candidates[w.rival]


def test_crops_contained_in_truth_and_rival_sometimes_matches():
    rival_hits = 0
    for seed in range(30):
        w = make(seed)
        for a in AGENTS:
            crop = w.crop_rows(a)
            assert len(crop) == 3 and all(len(r) == 4 for r in crop)
            assert contains(w.candidates[w.truth], crop)
            names = candidates_containing(w.candidates, crop)
            assert w.truth in names
            rival_hits += w.rival in names
    assert rival_hits > 0


def test_observe_round_trip_and_private():
    w = make(2)
    obs = w.observe(AGENTS[0])
    assert len(obs.parts) == 1
    cands, crop = parse_observation(obs.parts[0].text)
    assert cands == w.candidates
    assert crop == w.crop_rows(AGENTS[0])
    y, x = w.crops[AGENTS[0]]
    assert obs.private == {"crop_y": y, "crop_x": x}
    assert list(cands) == sorted(cands)


def test_numbers_names_round_trip():
    w = make(2, candidate_names="numbers", n_candidates=5)
    assert list(w.candidates) == ["1", "2", "3", "4", "5"]
    cands, _ = parse_observation(w.observe(AGENTS[1]).parts[0].text)
    assert cands == w.candidates


def test_guess_updates_status_and_score():
    w = make(4, agents=AGENTS[:4])
    other = next(n for n in w.candidates if n != w.truth)
    assert w.score() == {"accuracy": 0.0, "n_guessed": 0, "truth": w.truth}
    outs = commit_guesses(w, {AGENTS[0]: w.truth, AGENTS[1]: other, AGENTS[2]: w.truth})
    assert all(o.accepted and o.feedback == {"recorded": True} for o in outs)
    assert outs[0].action_id == "r0_a000"
    assert w.my_status(AGENTS[0]) == {"current_guess": w.truth, "guesses_made": 1}
    assert w.my_status(AGENTS[3]) == {"current_guess": None, "guesses_made": 0}
    cs = w.collective_status()
    assert cs["agents_with_guess"] == 3
    assert cs["guess_counts"][w.truth] == 2 and cs["guess_counts"][other] == 1
    assert set(cs["guess_counts"]) == set(w.candidates)
    assert w.score() == {"accuracy": 0.5, "n_guessed": 3, "truth": w.truth}
    commit_guesses(w, {AGENTS[0]: other}, 1)  # latest guess counts
    assert w.my_status(AGENTS[0]) == {"current_guess": other, "guesses_made": 2}
    assert w.score()["accuracy"] == 0.25


def test_validate():
    w = make(5, guess_limit=2)
    a = AGENTS[0]
    good = Action(name="guess", args={"candidate": "A"})
    assert w.validate(a, good).ok
    assert not w.validate(a, Action(name="guess", args={"candidate": "Z"})).ok
    assert not w.validate(a, Action(name="guess", args={})).ok
    assert not w.validate(a, Action(name="nope", args={})).ok
    assert not w.validate(AgentId("a999"), good).ok
    commit_guesses(w, {a: "A"})
    assert w.validate(a, good).ok
    commit_guesses(w, {a: "B"})
    ack = w.validate(a, good)
    assert not ack.ok and "limit" in ack.error
    # commit path re-checks the limit (guesses validated against round-start state)
    out = commit_guesses(w, {a: "C"})[0]
    assert not out.accepted
    assert w.my_status(a) == {"current_guess": "B", "guesses_made": 2}


def test_status_tools_disabled():
    w = make(6, status_tools=())
    assert [s.name for s in w.tool_schemas()] == ["guess"]
    assert w.my_status(AGENTS[0]) is None and w.collective_status() is None
    w2 = make(6, status_tools=["collective_status"])
    assert [s.name for s in w2.tool_schemas()] == ["guess", "collective_status"]
    full = make(6)
    assert [s.name for s in full.tool_schemas()] == ["guess", "my_status", "collective_status"]


def test_spec():
    w = FlagGame(n_candidates=6)
    assert w.spec()["type"] == "flaggame"
    assert w.spec()["params"]["n_candidates"] == 6
    assert w.spec()["params"]["status_tools"] == ["my_status", "collective_status"]


def test_snapshot_restore_mid_game():
    w = make(9)
    commit_guesses(w, {AGENTS[0]: "A", AGENTS[1]: "C"})
    blob = w.snapshot()
    fresh = FlagGame()
    fresh.restore(blob)
    assert fresh.verify() == w.verify()
    assert fresh.score() == w.score()
    assert fresh.collective_status() == w.collective_status()
    assert fresh.observe(AGENTS[3]) == w.observe(AGENTS[3])
    commit_guesses(w, {AGENTS[2]: "B"}, 1)
    commit_guesses(fresh, {AGENTS[2]: "B"}, 1)
    assert pickle.loads(fresh.snapshot()) == pickle.loads(w.snapshot())
    # config is not part of the snapshot: a restored world keeps its own constructor config
    limited = FlagGame(guess_limit=1)
    limited.restore(blob)
    assert limited.guess_limit == 1


def _bare(name, text):
    return re.search(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", text) is not None


@pytest.mark.parametrize("seed", range(3))
def test_oracle_no_leak(seed):
    agents = AGENTS[:8]
    w = make(seed, agents=agents)
    names = list(w.candidates)
    truth = w.truth
    obs0 = {a: w.observe(a).model_dump_json() for a in agents}
    schemas = json.dumps([s.model_dump() for s in w.tool_schemas()])
    dumps = [schemas]
    made = {a: 0 for a in agents}
    for rnd in range(20):
        guesses = {a: names[(rnd + i) % len(names)] for i, a in enumerate(agents)}
        acks = [w.validate(a, Action(name="guess", args={"candidate": c})) for a, c in guesses.items()]
        assert all(ack.ok and ack.error is None for ack in acks)
        outs = commit_guesses(w, guesses, rnd)
        for out, a in zip(outs, agents):
            made[a] += 1
            # the outcome is identical whether or not the guess was right
            assert out.model_dump(exclude={"action_id"}) == {"accepted": True, "feedback": {"recorded": True}}
            # my_status only echoes the agent's own guess
            assert w.my_status(a) == {"current_guess": guesses[a], "guesses_made": made[a]}
            assert w.observe(a).model_dump_json() == obs0[a]
        cs = w.collective_status()
        expected = {n: 0 for n in names}
        for c in guesses.values():
            expected[c] += 1
        assert cs == {"guess_counts": expected, "agents_with_guess": len(agents)}
        assert list(cs["guess_counts"]) == names
        dumps += [str(ack) for ack in acks] + [str(o) for o in outs]
        dumps += [str(w.my_status(a)) for a in agents] + [str(cs)]
    for d in dumps + list(obs0.values()):
        low = d.lower()
        assert "correct" not in low and "truth" not in low
    # truth name only where every candidate name appears symmetrically
    assert not _bare(truth, schemas)
    for a in agents:
        text = w.observe(a).parts[0].text
        assert all(text.count(f"\n{n}:\n") == 1 for n in names)
        stripped = text
        for n in names:
            stripped = stripped.replace(f"\n{n}:\n", "\n")
        assert not _bare(truth, stripped)
    for d in [str(ack) for ack in acks] + [str(o) for o in outs]:
        assert not _bare(truth, d)
