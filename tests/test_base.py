import random

import pytest

from swarmlab import (
    Action,
    Outcome,
    Participant,
    Persistable,
    Plugin,
    World,
    text_observation,
    tool,
)
from swarmlab.base import _plain
from swarmlab.ids import ActionId, AgentId


class Counter(World):
    def reset(self, rng, agents):
        self.total = 0

    def observe(self, agent):
        return text_observation(f"total so far: {self.total}")

    @tool("add", "Add n to the shared total", {"n": "integer"})
    def add(self, agent, n: int) -> Outcome:
        self.total += n
        return Outcome(accepted=True, feedback={"added": n})

    def score(self):
        return {"total": self.total}


class Leaf(Plugin):
    def __init__(self, a, b=2, *, c="x"):
        self.a = a


class Branch(Plugin):
    entry_point = "branch"

    def __init__(self, child, children=(), weight=1.0):
        self.child = child


class Sub(Leaf):
    def __init__(self, z=9):
        super().__init__(a=1)


class Bag(Persistable):
    def __init__(self):
        self.params = {"k": 1}
        self.items = [1, 2]
        self.rng = random.Random(5)


class Dummy(Participant):
    async def turn(self, view, tools):
        raise NotImplementedError


# ---- Persistable ------------------------------------------------------------------------------


def test_persistable_round_trip():
    b = Bag()
    b.rng.random()
    blob = b.snapshot()
    expected_next = b.rng.random()
    b.items.append(3)
    b.restore(blob)
    assert b.items == [1, 2]
    assert b.rng.random() == expected_next


def test_persistable_skips_params():
    b = Bag()
    blob = b.snapshot()
    b.params = {"k": 2}
    b.restore(blob)
    assert b.params == {"k": 2}  # params are constructor config, not state


def test_participant_bind_and_snapshot():
    p = Dummy()
    p.bind(AgentId("a003"), random.Random(1))
    blob = p.snapshot()
    q = Dummy()
    q.restore(blob)
    assert q.agent == "a003"
    assert q.rng.random() == random.Random(1).random()


# ---- Plugin.spec ------------------------------------------------------------------------------


def test_spec_captures_kwargs_and_defaults():
    leaf = Leaf(5)
    assert leaf.params == {"a": 5, "b": 2, "c": "x"}
    assert leaf.spec() == {"type": "tests.test_base:Leaf", "params": {"a": 5, "b": 2, "c": "x"}}
    assert Leaf(a=1, b=3, c="y").spec()["params"] == {"a": 1, "b": 3, "c": "y"}


def test_spec_nested_plugins_serialise():
    br = Branch(Leaf(1), children=(Leaf(2, c="q"),))
    assert br.spec() == {
        "type": "branch",
        "params": {
            "child": {"type": "tests.test_base:Leaf", "params": {"a": 1, "b": 2, "c": "x"}},
            "children": [{"type": "tests.test_base:Leaf", "params": {"a": 2, "b": 2, "c": "q"}}],
            "weight": 1.0,
        },
    }


def test_spec_most_derived_constructor_wins():
    assert Sub(z=4).params == {"z": 4}


def test_spec_without_init_is_empty_params():
    assert Counter().spec() == {"type": "tests.test_base:Counter", "params": {}}


def test_plain_handles_dicts_and_tuples():
    assert _plain({"x": (1, Leaf(0))}) == {
        "x": [1, {"type": "tests.test_base:Leaf", "params": {"a": 0, "b": 2, "c": "x"}}]
    }


# ---- World defaults ---------------------------------------------------------------------------


def make_counter() -> Counter:
    w = Counter()
    w.reset(random.Random(0), [AgentId("a000"), AgentId("a001")])
    return w


def test_tool_decorator_schema():
    schemas = make_counter().tool_schemas()
    assert [s.name for s in schemas] == ["add"]  # no status tools by default
    s = schemas[0]
    assert s.description == "Add n to the shared total"
    assert s.parameters["properties"] == {"n": {"type": "integer"}}
    assert s.parameters["required"] == ["n"]


def test_tool_rejects_unknown_json_type():
    with pytest.raises(ValueError):
        tool("x", "bad", {"n": "int"})


def test_default_validate():
    w = make_counter()
    assert w.validate(AgentId("a000"), Action(name="add", args={"n": 1})).ok
    bad = w.validate(AgentId("a000"), Action(name="mul", args={"n": 1}))
    assert not bad.ok and "unknown action" in bad.error
    assert not w.validate(AgentId("a000"), Action(name="add", args={})).ok
    assert not w.validate(AgentId("a000"), Action(name="add", args={"n": 1, "m": 2})).ok


def test_default_commit_applies_in_order():
    w = make_counter()
    acts = [
        (AgentId("a001"), ActionId("x1"), Action(name="add", args={"n": 2})),
        (AgentId("a000"), ActionId("x2"), Action(name="add", args={"n": 5})),
        (AgentId("a000"), ActionId("x3"), Action(name="nope", args={})),
    ]
    outs = w.commit(acts)
    assert [o.action_id for o in outs] == ["x1", "x2", "x3"]
    assert [o.accepted for o in outs] == [True, True, False]
    assert outs[1].feedback == {"added": 5}
    assert w.score() == {"total": 7}
    assert w.observe(AgentId("a000")).parts[0].text == "total so far: 7"


def test_world_defaults_and_snapshot():
    w = make_counter()
    assert w.terminal() is False and w.verify() == {}
    assert w.my_status(AgentId("a000")) is None and w.collective_status() is None
    w.commit([(AgentId("a000"), ActionId("x"), Action(name="add", args={"n": 3}))])
    blob = w.snapshot()
    w.commit([(AgentId("a000"), ActionId("y"), Action(name="add", args={"n": 3}))])
    w.restore(blob)
    assert w.score() == {"total": 3}


class StatusCounter(Counter):
    def my_status(self, agent):
        return {"total": self.total}


def test_status_tool_exposed_when_enabled():
    w = StatusCounter()
    w.reset(random.Random(0), [])
    assert [s.name for s in w.tool_schemas()] == ["add", "my_status"]
