"""`@tool` with callable `params`/`description`: schemas that follow the world instance."""
import random

import pytest

from swarmlab import Action, Experiment, Outcome, Participant, World, text_observation, tool
from swarmlab.ids import AgentId
from swarmlab.participants.llm import LLMAgent
from swarmlab.prompts_cmd import group_views, render_prompts

A = AgentId("a000")


class Picker(World):
    """`pick(x)` with `x` an enum of range(n); `tag` adds a second argument only when n > 2."""

    def __init__(self, n: int = 3):
        self.n = n

    def reset(self, rng, agents):
        self.picks: list[int] = []

    def observe(self, agent):
        return text_observation(f"picks: {self.picks}")

    @tool("pick", lambda self: f"Pick one of {self.n} numbers",
          {"x": lambda self: {"type": "integer", "enum": list(range(self.n))}})
    def pick(self, agent, x: int) -> Outcome:
        self.picks.append(x)
        return Outcome(accepted=True, feedback={"x": x})

    @tool("tag", "Tag something", lambda self: {"label": "string", **({"note": "string"} if self.n > 2 else {})})
    def tag(self, agent, **args) -> Outcome:
        return Outcome(accepted=True)

    def score(self):
        return {"picks": list(self.picks)}


def make(n: int) -> Picker:
    w = Picker(n)
    w.reset(random.Random(0), [A])
    return w


def test_schema_follows_the_instance():
    s = make(4).tool_schema("pick")
    assert s.description == "Pick one of 4 numbers"
    assert s.parameters["properties"] == {"x": {"type": "integer", "enum": [0, 1, 2, 3]}}
    assert s.parameters["required"] == ["x"] and s.parameters["additionalProperties"] is False
    assert {t.name: t for t in make(4).tool_schemas()}["pick"] == s


def test_two_instances_differ():
    two, five = make(2).tool_schemas(), make(5).tool_schemas()
    assert two != five
    assert two[0].parameters["properties"]["x"]["enum"] == [0, 1]
    assert five[0].parameters["properties"]["x"]["enum"] == [0, 1, 2, 3, 4]
    assert two[1].parameters["required"] == ["label"]
    assert five[1].parameters["required"] == ["label", "note"]


def test_validate_uses_the_dynamic_required_list():
    small, big = make(2), make(3)
    assert small.validate(A, Action(name="tag", args={"label": "a"})).ok
    assert not small.validate(A, Action(name="tag", args={"label": "a", "note": "b"})).ok
    assert not big.validate(A, Action(name="tag", args={"label": "a"})).ok
    assert big.validate(A, Action(name="tag", args={"label": "a", "note": "b"})).ok


def test_callable_unknown_json_type_raises_at_schema_time():
    class Bad(Picker):
        @tool("bad", "Bad", {"y": lambda self: "int"})
        def bad(self, agent, y) -> Outcome:
            return Outcome(accepted=True)

    class BadWhole(Picker):
        @tool("bad", "Bad", lambda self: {"y": "int"})
        def bad(self, agent, y) -> Outcome:
            return Outcome(accepted=True)

    for cls in (Bad, BadWhole):  # decoration succeeded; building the schema fails
        with pytest.raises(ValueError, match="unknown JSON type 'int'"):
            cls().tool_schemas()
    with pytest.raises(ValueError):  # static parts are still checked at decoration time
        tool("x", "bad", {"a": "int", "b": lambda self: "string"})


class PickLast(Participant):
    """Picks the largest value the schema it is shown allows."""

    async def turn(self, view, tools):
        (s,) = [t for t in tools.schemas() if t.name == "pick"]
        await tools.call("pick", {"x": s.parameters["properties"]["x"]["enum"][-1]})


def test_experiment_run_and_prompts_show_the_instance_enum(tmp_path):
    run = Experiment(name="pick", world=Picker(n=6), participants=[PickLast()] * 2).run(
        seed=0, max_rounds=2, out=tmp_path)
    assert run.score["picks"] == [5, 5, 5, 5]

    exp = Experiment(name="pick", world=Picker(n=6),
                     participants=[LLMAgent("fake:reader", tool_protocol="json")])
    (_, _, _, _, view), = group_views(exp)
    assert {t.name: t for t in view.tools}["pick"].parameters["properties"]["x"]["enum"] == [0, 1, 2, 3, 4, 5]
    (row,) = render_prompts(exp)
    assert '"enum": [0, 1, 2, 3, 4, 5]' in row["system"]
