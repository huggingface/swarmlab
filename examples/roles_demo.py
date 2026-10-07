"""M3c: roles and a two-level hierarchy in Python (the YAML twin is hierarchy_flaggame.yaml).

18 LLM agents on the Flag Game, two arms: `flat` (broadcast, all workers) and `tree` (a Tree
topology of 3 groups of 6; the first agent of each group is its coordinator). It also defines a
custom role, `scout`: a worker whose prompt asks it to report its evidence verbatim and whose turn
is capped at 6 tool calls and 200 output tokens; one scout per group.

Runs on the deterministic fake provider (`fake:reader`: no network, no spend) by default. For a
real model set MODEL = "anthropic:claude-haiku-4-5" (needs ANTHROPIC_API_KEY; check
`exp.estimate(...)` and the budget first).

    uv run python examples/roles_demo.py
"""
from swarmlab import Board, Budget, Experiment
from swarmlab.medium.topology import Tree
from swarmlab.participants import LLMAgent
from swarmlab.prompts_cmd import render_prompts
from swarmlab.roles import COORDINATOR, WORKER, Role, assign
from swarmlab.worlds import FlagGame

MODEL = "fake:reader"  # or "anthropic:claude-haiku-4-5"
N, GROUPS = 18, 3

SCOUT = Role(
    name="scout",
    prompt_append="When you post, quote your crop rows verbatim first, then your best guess.",
    budget={"max_calls": 6, "max_tokens": 200},
)


def agent(role: str) -> LLMAgent:
    return assign(LLMAgent(model=MODEL, max_tokens=256, max_calls=4), role)


def experiment(arm: str, coordinators_act: bool = False) -> Experiment:
    if arm == "flat":
        participants = [agent("worker") for _ in range(N)]
        medium = Board(topology="broadcast")
    else:
        size = N // GROUPS
        participants = [agent("coordinator" if i % size == 0 else "scout" if i % size == 1 else "worker")
                        for i in range(N)]
        medium = Board(topology=Tree(groups=GROUPS))
    return Experiment(
        name="roles-demo", arm=arm, world=FlagGame(n_candidates=8, rival_edits=1),
        participants=participants, medium=medium,
        roles={"worker": WORKER, "scout": SCOUT,
               # the ablation switch: coordinators that may also guess
               "coordinator": COORDINATOR.model_copy(update={"may_act": coordinators_act})},
        metrics=["belief.consensus", "belief.accuracy", "comm.read_rate"],
        budget=Budget(hard_usd=1.0),  # nominal: the fake provider bills nothing
    )


if __name__ == "__main__":
    tree = experiment("tree")
    scout = next(r for r in render_prompts(tree, seed=1) if r["role"] == "scout")
    print("scout system prompt ends with:", scout["system"].splitlines()[-1])
    for arm in ("flat", "tree"):
        run = experiment(arm).run(seed=1, max_rounds=4, out="runs")
        roles = sorted({(e["agent"], e["role"]) for e in run.events if e["type"] == "turn_started"})
        print(f"{arm}: score={run.score} roles={dict(roles[:GROUPS * 2])}...")
    print("ablation changes the spec hash:",
          experiment("tree").spec_hash(1, 4) != experiment("tree", coordinators_act=True).spec_hash(1, 4))
