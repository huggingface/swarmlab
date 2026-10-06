"""M1b: LLM agents on the Flag Game with a belief probe every round.

Runs on the deterministic fake provider (`fake:reader`: no network, no spend) by default. To use
Claude Haiku 4.5 instead, set MODEL = "anthropic:claude-haiku-4-5" (needs ANTHROPIC_API_KEY or
ANTHROPIC_KEY; the budget below caps the spend).

    uv run python examples/04_llm_flaggame.py
"""
from swarmlab import Board, Budget, Experiment
from swarmlab.metrics.belief import Accuracy, Consensus
from swarmlab.participants import LLMAgent
from swarmlab.probes import BeliefProbe
from swarmlab.worlds import FlagGame

MODEL = "fake:reader"  # or "anthropic:claude-haiku-4-5"

exp = Experiment(
    name="flag-llm",
    world=FlagGame(n_candidates=8),
    participants=[LLMAgent(model=MODEL, max_tokens=512, max_calls=6)] * 4,
    medium=Board(topology="broadcast"),
    metrics=["belief.consensus", "belief.accuracy",
             Consensus(source="probe:belief"), Accuracy(source="probe:belief")],
    probes=[BeliefProbe()],
    budget=Budget(soft_usd=0.40, hard_usd=0.50, measurement_usd=0.10),
)

if __name__ == "__main__":
    print("estimate:", exp.estimate(seed=1, max_rounds=3)["usd"])
    run = exp.run(seed=1, max_rounds=3)
    print("score:", run.score)
    print("spend:", run.spend)
    for name, values in run.metrics.items():
        print(f"{name}:", values)
    print("probes:", run.probes["belief"][:4])
    print("view:", run.view())
