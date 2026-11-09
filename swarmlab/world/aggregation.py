"""Information aggregation (s3, arXiv 2611.08615, §4.2 and Appendices F-H).

Each of the N agents holds one private integer from 1 to 99, shown as English words
("seventy-three"). The team must return all N numbers in ascending order, keeping equal numbers
from different agents as separate entries. It succeeds if any agent submits the full, correct
list. No agent can succeed alone at N >= 2, so the task tests whether agents share and collect
private information.

Per run: 16 numbers drawn uniformly from 1..99, of which the first N are used (the paper's
evaluation problems hold sixteen, so one problem serves every team size), and the agents' public
names, N distinct names sampled from `agent_1` .. `agent_64`, each agent seeing
its teammates in its own shuffled order (the paper samples fresh names and orders per episode so
no convention can attach to a name). With `problem: <int>` the numbers come from that problem id
alone (`random.Random(f"aggregation-problem-{problem}")`), so several runs (the paper's G team
attempts per problem) share the numbers while names and order still come from the run seed;
without it (default) the numbers come from the world rng.

The observation has two text parts: the paper's first user turn (`FIRST_USER_TURN`, Appendix G),
then a roster line, `ROSTER_PREFIX` followed by JSON `{"name", "teammates", "ids", "n"}`: the
agent's public name, its teammates' names in its shuffled order, and name -> swarmlab agent id. The
roster is for the harness (the `s3_agent` participant renders the system prompt from it and maps
names to ids when posting); it is not shown to the model as is. `private` holds the number.
`description()` is empty.

`submit(answer)` records an agent's final answer (a list of integers) once; the feedback never
says whether it is right. Score: `success` (any submitted answer equals the sorted numbers),
`n_answered`, `n_correct`, `numbers`, `answers`. The run is `terminal` when every agent has
submitted. Requires `commit: async` (agents act concurrently, without rounds).
"""
from __future__ import annotations

import random
import json
from typing import Any, ClassVar

from ..ids import AgentId
from ..view import Observation, Part
from .base import Outcome, Requirements, World, tool

N_NUMBERS = 16  # numbers per problem; the first N are used
NAME_POOL = 64  # public names agent_1 .. agent_64
ROSTER_PREFIX = "Roster: "

_ONES = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
         "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_words(n: int) -> str:
    """1..99 in English words, hyphenated like the paper's examples ("forty-two")."""
    if not 1 <= n <= 99:
        raise ValueError(f"number must be in 1..99, got {n}")
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")


def count_words(n: int) -> str:
    """A team size in words, for the prompt texts written for N = 4 ("four")."""
    return _ONES[n] if 1 <= n < 20 else str(n)


# Appendix G, verbatim at N = 4 ("four"); `{n}` is the team size in words.
FIRST_USER_TURN = (
    "Solve this problem with your team: {n_cap} agents each have one private number, written in English. "
    "Find all {n} assigned numbers through communication and return their numeric values in ascending "
    "order. Include duplicates once per owning agent. You know only your own assignment; other agents do "
    "not know it unless you tell them. Final format: #### followed by a JSON array of {n} integers. "
    "Your private number: {number}."
)


class Aggregation(World):
    """Private-information number sorting: share one number each, any agent submits the sorted list."""

    entry_point: ClassVar[str | None] = "aggregation"
    requires: ClassVar[Requirements] = Requirements(commit="async")
    name = "aggregation"
    _skip_in_snapshot: ClassVar[tuple[str, ...]] = ("params", "problem")

    def __init__(self, problem: int | None = None) -> None:
        if problem is not None and (isinstance(problem, bool) or not isinstance(problem, int)):
            raise ValueError(f"problem must be an integer, got {problem!r}")
        self.problem = problem
        self.agents: list[AgentId] = []
        self.numbers: dict[str, int] = {}
        self.names: dict[str, str] = {}             # agent id -> public name
        self.teammates: dict[str, list[str]] = {}   # agent id -> teammate names, shuffled
        self.answers: dict[str, list[int] | None] = {}

    def reset(self, rng: random.Random, agents: list[AgentId]) -> None:
        self.agents = list(agents)
        if len(self.agents) > N_NUMBERS:
            raise ValueError(f"at most {N_NUMBERS} agents, got {len(self.agents)}")
        src = rng if self.problem is None else random.Random(f"aggregation-problem-{self.problem}")
        drawn = [src.randint(1, 99) for _ in range(N_NUMBERS)]
        self.numbers = {str(a): drawn[i] for i, a in enumerate(self.agents)}
        names = rng.sample([f"agent_{i}" for i in range(1, NAME_POOL + 1)], len(self.agents))
        self.names = {str(a): names[i] for i, a in enumerate(self.agents)}
        self.teammates = {}
        for a in self.agents:
            others = [self.names[str(b)] for b in self.agents if b != a]
            rng.shuffle(others)
            self.teammates[str(a)] = others
        self.answers = {}

    def target(self) -> list[int]:
        return sorted(self.numbers.values())

    def description(self) -> str:
        return ""  # the s3_agent participant renders the paper's system prompt itself

    def observe(self, agent: AgentId) -> Observation:
        n = count_words(len(self.agents))
        text = FIRST_USER_TURN.format(n_cap=n.capitalize(), n=n,
                                      number=number_words(self.numbers[str(agent)]))
        roster = {"name": self.names[str(agent)], "teammates": self.teammates[str(agent)],
                  "ids": {self.names[str(b)]: str(b) for b in self.agents}, "n": len(self.agents)}
        return Observation(parts=[Part(type="text", text=text),
                                  Part(type="text", text=ROSTER_PREFIX + json.dumps(roster))],
                           private={"number": self.numbers[str(agent)]})

    @tool("submit", "Submit the team's sorted list of numbers (your final answer).",
          {"answer": {"type": "array", "items": {"type": "integer"}}})
    def submit(self, agent: AgentId, answer: Any) -> Outcome:
        if str(agent) in self.answers:
            return Outcome(accepted=False, feedback={"error": "already submitted"})
        ok = isinstance(answer, list) and all(isinstance(v, int) and not isinstance(v, bool) for v in answer)
        self.answers[str(agent)] = list(answer) if ok else None
        return Outcome(accepted=True, feedback={"recorded": True})

    def terminal(self) -> bool:
        return bool(self.agents) and len(self.answers) == len(self.agents)

    def score(self) -> dict:
        target = self.target()
        correct = [a for a, ans in self.answers.items() if ans == target]
        return {"success": bool(correct), "n_agents": len(self.agents), "n_answered": len(self.answers),
                "n_correct": len(correct), "numbers": dict(self.numbers), "names": dict(self.names),
                "answers": dict(self.answers)}

    def verify(self) -> dict:
        return {"numbers": dict(self.numbers), "names": dict(self.names)}
