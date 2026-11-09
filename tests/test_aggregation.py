from __future__ import annotations

import json

from swarmlab.ids import agent_id
from swarmlab.participants.s3 import after_thinking, parse_answer
from swarmlab.rng import derive
from swarmlab.world.aggregation import ROSTER_PREFIX, Aggregation
from swarmlab.world.base import Action


def world(seed=1, problem=None, n=4):
    w = Aggregation(problem=problem)
    w.reset(derive(seed, "world"), [agent_id(i) for i in range(n)])
    return w


def test_problem_fixes_numbers_while_names_and_roster_vary_by_seed():
    a, b = world(1, problem=7), world(2, problem=7)
    assert a.numbers == b.numbers and a.names != b.names  # G team attempts share one problem
    assert world(1, problem=8).numbers != a.numbers
    roster = json.loads(a.observe(agent_id(0)).parts[1].text[len(ROSTER_PREFIX):])
    # the harness posts by name: every teammate name must map to that teammate's agent id
    assert {roster["ids"][t] for t in roster["teammates"]} == {"a001", "a002", "a003"}
    assert roster["ids"][roster["name"]] == "a000" and roster["name"] not in roster["teammates"]


def test_any_correct_answer_succeeds_and_feedback_hides_correctness():
    w = world()
    out = w.commit([(agent_id(0), "x0", Action(name="submit", args={"answer": [1, 2, 3, 4]})),
                    (agent_id(1), "x1", Action(name="submit", args={"answer": w.target()})),
                    (agent_id(1), "x2", Action(name="submit", args={"answer": w.target()}))])
    assert [o.accepted for o in out] == [True, True, False]  # one answer per agent
    assert out[0].feedback == out[1].feedback == {"recorded": True}
    s = w.score()
    assert s["success"] and s["n_correct"] == 1 and not w.terminal()


def test_final_answer_is_read_after_the_thinking_only():
    text = "<think>maybe #### [9, 9, 9, 9]</think>#### [3, 1] then #### [1, 2, 3, 4]"
    assert parse_answer(after_thinking(text)) == [1, 2, 3, 4]
    assert parse_answer(after_thinking("<think>#### [1, 2, 3, 4]</think>[1, 2, 3, 4]")) is None  # no ####
