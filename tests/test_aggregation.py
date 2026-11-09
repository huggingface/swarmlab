from __future__ import annotations

from swarmlab.ids import agent_id
from swarmlab.participants.s3 import after_thinking, parse_answer, system_prompt
from swarmlab.rng import derive
import json

from swarmlab.world.aggregation import ROSTER_PREFIX, Aggregation, number_words
from swarmlab.world.base import Action


def world(n=4, seed=1):
    w = Aggregation()
    w.reset(derive(seed, "world"), [agent_id(i) for i in range(n)])
    return w


def test_number_words():
    assert [number_words(n) for n in (1, 13, 20, 42, 99)] == ["one", "thirteen", "twenty", "forty-two",
                                                             "ninety-nine"]


def test_first_user_turn_and_private_names():
    w = world()
    obs = w.observe(agent_id(0))
    text = obs.parts[0].text
    assert text.startswith("Solve this problem with your team: Four agents each have one private number")
    assert text.endswith(f"Your private number: {number_words(w.numbers['a000'])}.")
    p = json.loads(obs.parts[1].text[len(ROSTER_PREFIX):])
    assert obs.private == {"number": w.numbers["a000"]}
    assert p["n"] == 4 and p["name"] not in p["teammates"] and len(p["teammates"]) == 3
    assert {p["ids"][t] for t in p["teammates"]} == {"a001", "a002", "a003"}
    assert world(seed=1).numbers == w.numbers and world(seed=2).names != w.names


def test_any_correct_answer_succeeds_and_feedback_hides_correctness():
    w = world()
    out = w.commit([(agent_id(0), "x0", Action(name="submit", args={"answer": [1, 2, 3, 4]})),
                    (agent_id(1), "x1", Action(name="submit", args={"answer": w.target()}))])
    assert all(o.feedback == {"recorded": True} for o in out)
    s = w.score()
    assert s["success"] and s["n_correct"] == 1 and not w.terminal()


def test_prompt_parts_and_answer_parsing():
    sp = system_prompt("agent_7", ["agent_3", "agent_50"], 4)
    assert sp.startswith("You are in a team solving a private-information number sorting task. Each of four")
    assert "Your name is agent_7. Your teammates: agent_3, agent_50." in sp
    assert parse_answer("x #### [3, 1] y #### [1, 2, 3]") == [1, 2, 3]
    assert parse_answer("#### [1, 'a']") is None
    assert after_thinking("<think>#### [9]</think>#### [1]") == "#### [1]"


def test_problem_fixes_the_numbers_not_the_names():
    def w(seed, problem):
        x = Aggregation(problem=problem)
        x.reset(derive(seed, "world"), [agent_id(i) for i in range(4)])
        return x
    a, b = w(1, 7), w(2, 7)
    assert a.numbers == b.numbers and a.names != b.names
    assert w(1, 8).numbers != a.numbers
