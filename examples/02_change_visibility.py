"""DESIGN.md example 2: change one scientific mechanism, here message visibility.

The `...` of the design doc is filled in with example 1's setup on a broadcast board.

    uv run python examples/02_change_visibility.py
"""
from swarmlab import Board, Budget, Experiment, Policy
from swarmlab.participants import EvidenceAggregator
from swarmlab.worlds import FlagGame


class OddAgentsSeeNothing(Policy):
    def apply(self, reader, post, round):
        if int(reader[1:]) % 2:
            return None                      # withhold
        return round + 1, post.text          # available next round, unchanged


exp = Experiment(
    name="flag-odd-blind",
    world=FlagGame(n_candidates=8),
    participants=[EvidenceAggregator()] * 16,
    medium=Board(topology="broadcast", policies=[OddAgentsSeeNothing()]),
    metrics=["belief.consensus", "belief.accuracy", "comm.read_rate"],
    budget=Budget(soft_usd=0, hard_usd=0),
)

if __name__ == "__main__":
    run = exp.run(seed=3, max_rounds=20)
    recipients = sorted({e["recipient"] for e in run.events if e["type"] == "delivery"})
    print("agents that received anything:", recipients)
    print("score:", run.score)
    print("view:", run.view())
