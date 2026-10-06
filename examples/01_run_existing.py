"""DESIGN.md example 1: run an existing task with built-in agents and communication.

    uv run python examples/01_run_existing.py
"""
from swarmlab import Board, Budget, Experiment, Run
from swarmlab.participants import EvidenceAggregator
from swarmlab.worlds import FlagGame

exp = Experiment(
    name="flag-gossip",
    world=FlagGame(n_candidates=8),
    participants=[EvidenceAggregator()] * 16,
    medium=Board(topology="gossip"),
    metrics=["belief.consensus", "belief.accuracy", "comm.read_rate"],
    budget=Budget(soft_usd=0, hard_usd=0),      # scripted agents spend nothing
)

if __name__ == "__main__":
    run = exp.run(seed=3, max_rounds=20)
    print("score:", run.score)                                  # final evaluator-side score
    print("consensus:", run.metrics["belief.consensus"][-3:])   # per-round (round, value, denominator)
    print("events:", sum(1 for _ in run.events))                # the logical event trajectory
    print("view:", run.view())                                  # builds and returns the replay page
    fork = run.fork(at_round=4).run()                           # live fork
    print("fork:", fork.id, fork.score)
    print("replayed:", Run.load(run.dir).score)           # Run.load(path) replays from disk
    for r in exp.run_all([4, 5], max_rounds=20):                # several seeds; skips existing runs
        s = r.summary()
        print(s["run_id"], s["end_reason"], s["score"])
