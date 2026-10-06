"""B1: a participant gets an agent-bound handle, never the round executor."""
import inspect

from swarmlab import AgentTools, Experiment, Participant, TurnUsage
from swarmlab.medium.board import Board
from swarmlab.world.flaggame import FlagGame
from tests.test_executor import AGENTS, setup


def test_agent_tools_exposes_no_executor_world_or_board(tmp_path):
    _, _, _, ex = setup(tmp_path)
    tools = AgentTools(ex, AGENTS[0])
    public = {n for n in dir(tools) if not n.startswith("_")}
    assert public == {"agent", "schemas", "call"}
    for name in ("world", "board", "executor", "blobs", "agents"):
        assert not hasattr(tools, name)
    assert not hasattr(tools, "__dict__")  # slots only: nothing can be attached by accident
    params = list(inspect.signature(tools.call).parameters)
    assert params == ["name", "args"]  # no agent parameter: the handle carries the identity
    assert list(inspect.signature(tools.schemas).parameters) == []
    assert tools.agent == AGENTS[0]
    assert "executor" not in repr(tools).lower()


async def test_agent_tools_acts_only_as_its_agent(tmp_path):
    _, _, _, ex = setup(tmp_path)
    tools = AgentTools(ex, AGENTS[0])
    res = await tools.call("guess", {"candidate": "A"})
    assert res.ok and res.result["id"].startswith(f"x0001-{AGENTS[0]}-")
    assert ex.buffered_actions(AGENTS[0]) and not ex.buffered_actions(AGENTS[1])
    assert [s.name for s in tools.schemas()] == [s.name for s in ex.schemas(AGENTS[0])]


class Impostor(Participant):
    """Tries every route to act as another agent; records what it found."""

    async def turn(self, view, tools):
        self.found = sorted(n for n in ("world", "board", "executor") if hasattr(tools, n))
        await tools.call("guess", {"candidate": "A"})
        return TurnUsage(calls=1)


def test_runner_passes_agent_tools(tmp_path):
    exp = Experiment(name="imp", world=FlagGame(), medium=Board(), participants=[Impostor()] * 2)
    run = exp.run(seed=0, max_rounds=1, out=tmp_path)
    committed = [e for e in run.events if e["type"] == "action_committed"]
    assert sorted(e["agent"] for e in committed) == ["a000", "a001"]
    assert all(e["action_id"].split("-")[1] == e["agent"] for e in committed)


class Yielder(Participant):
    """Behaviour chosen by agent index: end_turn+usage, end_turn then late call, cap, error, no_tool."""

    async def turn(self, view, tools):
        i = int(self.agent[1:])
        if i == 0:
            await tools.call("end_turn", {})
            return TurnUsage(calls=1, prompt_tokens=7, completion_tokens=3, cost_usd=0.5)
        if i == 1:
            await tools.call("end_turn", {})
            self.late = (await tools.call("guess", {"candidate": "A"})).model_dump()
            return TurnUsage(calls=2)
        if i == 2:
            while True:
                await tools.call("my_status", {})
        if i == 3:
            raise RuntimeError("boom")
        return TurnUsage(calls=0)


def test_yield_kinds_and_usage_after_end_turn(tmp_path):
    exp = Experiment(name="yield", world=FlagGame(), medium=Board(), participants=[Yielder()] * 5)
    run = exp.run(seed=0, max_rounds=1, out=tmp_path, max_calls_per_turn=3)
    ended = {e["agent"]: e for e in run.events if e["type"] == "turn_ended"}
    assert {a: e["yield_kind"] for a, e in ended.items()} == {
        "a000": "end_turn", "a001": "end_turn", "a002": "cap", "a003": "error", "a004": "no_tool"}
    assert ended["a000"]["usage"] == {"calls": 1, "prompt_tokens": 7, "completion_tokens": 3,
                                      "cost_usd": 0.5}
    assert ended["a001"]["usage"]["calls"] == 2
    assert "RuntimeError: boom" in ended["a003"]["error"]
    # the guess after end_turn was rejected, logged, and never committed
    rets = [e for e in run.events if e["type"] == "tool_returned" and e["agent"] == "a001"]
    assert rets[-1]["result"]["error"] == "turn_ended"
    assert not [e for e in run.events if e["type"] == "action_committed"]


class PrivacyProbe(Participant):
    """A5: the view must not carry the world's evaluator-only data."""

    async def turn(self, view, tools):
        assert view.observation.private == {}
        assert "crop_y" not in view.model_dump_json()
        return TurnUsage()


def test_private_is_stripped_from_view_and_logged_on_turn_started(tmp_path):
    exp = Experiment(name="priv", world=FlagGame(), medium=Board(), participants=[PrivacyProbe()] * 3)
    run = exp.run(seed=4, max_rounds=2, out=tmp_path)
    ended = [e for e in run.events if e["type"] == "turn_ended"]
    assert ended and all(e["yield_kind"] == "no_tool" for e in ended), ended[0].get("error")
    started = [e for e in run.events if e["type"] == "turn_started"]
    world = FlagGame()
    from swarmlab.rng import derive

    world.reset(derive(4, "world"), ["a000", "a001", "a002"])
    for e in started:
        assert e["private"] == world.observe(e["agent"]).private
        assert set(e["private"]) == {"crop_y", "crop_x"}
