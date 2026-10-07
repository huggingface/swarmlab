"""`swarmlab prompts` shows the round-1 observation a run shows (field notes item 6)."""
import yaml

from swarmlab import Experiment, Outcome, World, text_observation, tool
from swarmlab.prompts_cmd import render_prompts


class Clock(World):
    """Observation depends on the round set by `begin_round`."""

    def reset(self, rng, agents):
        self.round = 0

    def begin_round(self, round):
        self.round = round

    def observe(self, agent):
        return text_observation(f"it is round {self.round} of the clock")

    @tool("tick", "Do nothing", {})
    def tick(self, agent) -> Outcome:
        return Outcome(accepted=True)

    def score(self):
        return {"round": self.round}


def test_prompts_call_begin_round_1(tmp_path):
    p = tmp_path / "clock.yaml"
    p.write_text(yaml.safe_dump({"name": "clock", "options": {"max_rounds": 1}, "arms": {"A": {
        "world": {"type": "tests.test_prompts_round:Clock"},
        "participants": [{"type": "llm", "params": {"model": "fake:reader"}}]}}}))
    exp = Experiment.from_yaml(p, "A")
    (row,) = render_prompts(exp, 0)
    assert "it is round 1 of the clock" in row["user"]  # not "round 0": no begin_round yet
    assert exp.world.score() == {"round": 0} or not hasattr(exp.world, "round")  # a copy
