"""DESIGN.md example 3: implement a new task with the smallest World interface.

`Counter` is verbatim from the design doc; `Adder` is a minimal scripted participant so the
world has someone to play it.

    uv run python examples/03_new_world.py
"""
from swarmlab import Experiment, Outcome, Participant, World, text_observation, tool


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


class Adder(Participant):
    """Adds `step` to the total every round."""

    def __init__(self, step: int = 1):
        self.step = step

    async def turn(self, view, tools):
        await tools.call(self.agent, "add", {"n": self.step})


exp = Experiment(name="count", world=Counter(), participants=[Adder(), Adder(step=2)] * 2)

if __name__ == "__main__":
    run = exp.run(seed=0, max_rounds=5)
    print("score:", run.score)                    # {'total': 30}
    print("view:", run.view())
