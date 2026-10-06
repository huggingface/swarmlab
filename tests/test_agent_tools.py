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
    world, _, _, ex = setup(tmp_path)
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
