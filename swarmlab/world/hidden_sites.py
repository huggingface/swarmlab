"""HiddenSites: hidden, fixed site qualities with renewable discoveries.

An allocation task with a strict information boundary. `n_sites` sites labelled A, B, C, ... have
hidden success probabilities that stay fixed for the whole run. Every round each site offers one
fresh discovery; at the end of the round every agent holding a site makes one independent
attempt there, and the team scores one point per site where at least one attempt
succeeded. More agents at a site raise its chance of scoring, with diminishing returns, so the
team does best by spreading over the good sites rather than crowding the best one. Agents learn
only their own attempts and the team's points; where the others are comes only through the board.

Constructor
-----------
`HiddenSites(probabilities=(0.60, 0.35, 0.25, 0.15, 0.10, 0.06, 0.03, 0.01), n_sites=8,
rounds=20, show_history=True, permute=True)`

- `probabilities`: the multiset of success probabilities, one per site, each in (0, 1).
- `n_sites`: 1..26 (labels A..Z); must equal `len(probabilities)`.
- `rounds`: the round count stated to agents ("Round r of R", the task text). It does not end
  the run: `terminal()` is always False and `max_rounds` ends it. `requires` makes the two agree
  (and `commit: immediate`), checked when the experiment is built and again at run start.
- `show_history`: adds the "Your attempts so far" line to the observation.
- `permute`: `reset` shuffles `probabilities` over the labels with the world rng (False: A gets
  `probabilities[0]`, B the next, ...).

Observation format (exact; `parse_observation` inverts it)
----------------------------------------------------------
One text part, `observation.private` empty:

    Round 2 of 20.
    Your current position: none | C
    Your previous attempt: none yet | site C, success | site C, failure
    Your attempts so far: none | C: 3 attempts, 1 success; A: 2 attempts, 0 successes
    Team points last round: not yet (points are scored after each full round) | 2
    Team points so far: 0 | 17

"Your attempts so far" (only with `show_history`) aggregates the agent's own resolved attempts
per site, in order of first attempt. "Your previous attempt" is the agent's attempt in the
previous round ("none yet" when it held no site then). Before `begin_round(1)` (as `swarmlab
prompts` renders it) the round shows as 1.

Actions
-------
One world tool, `choose_site(site)`, `site` an enum of the labels (callable `@tool` description and
param, so the schema follows `n_sites`; the description has no numbers). It moves the agent to `site` from
now on (choosing the site it already holds is allowed and counts as its choice). **One choice per
round**: a second call by the same agent in the same round is rejected with
`accepted=False, feedback={"error": "already_chose_this_round"}` and changes nothing; the set of
agents who chose is cleared by `begin_round`. An unknown site is rejected with
`{"error": "unknown_site", "detail"}` and does not use up the choice. Accepted feedback is
`{"position": <site>, "previous": <site or None>}` and nothing else. No status tools
(`my_status`, `collective_status` stay None). The rule is per round whatever the scheduler: with
the default scheduler (one turn per agent per round) it is one choice per turn; under `fixed`,
`one_speaker` or another scheduler it still allows one accepted choice per agent per round.

Information boundary
--------------------
Agents never see: the probabilities or their order, which site is best, how many agents hold any
site, other agents' positions or attempts, expected points or any reference allocation. They see
their own position, their own attempts and outcomes, and the team's points per resolved round and
in total. The task text (`description()`) has no numbers at all and is identical across seeds.

The task text is scheduler- and topology-neutral: it describes the task, the sites, the attempts,
the one-choice-per-round rule, what an agent is told after each round and that the others'
positions are learned only through the board. It says nothing about who takes turns, in which
order, or who can read a post, because the scheduler and the board topology are legitimate
variations that run with the same text. An experiment that wants agents told how its schedule and
delivery work says so in `system_prompt_append` (as `examples/hidden_sites.yaml` does).

Resolution and the trial stream
-------------------------------
Choosing a site is not an attempt. Round r is resolved from the END-of-round positions: every
assigned agent `a` at site `k` attempts once, and succeeds iff

    derive(trial_key, "trial", r, a, k).random() < p_k

with `trial_key` a 64-bit integer drawn from the world rng at reset (after the permutation). Every
counterfactual attempt is therefore fixed in advance by the seed: independent across
(round, agent, site), independent of tool-call counts, of the order of turns and of who else is
at the site. Two arms with the same seed (for example a control and a one-sentence prompt
treatment) get the same hidden assignment and the same coin for every possible attempt, so
their differences come from where agents stand, not from luck.

    R_r = number of sites with at least one success in round r          (actual points)
    W_r = sum_k [1 - (1 - p_k) ** n_k]  over the end-of-round counts n   (expected points)

`end_round(r)` (after the round's last commit) resolves round r from the positions then in force,
stores it (`allocations`, `attempts`, `points_by_round`, `expected_by_round`) and returns
`{"W", "R", "counts" {site: n}, "outcomes" {agent: [site, success]}}`, which the runner logs as a
`round_resolved` event. `score()` only reads the stored lists; the snapshot is taken after
`end_round`, so the last round is scored like every other and `swarmlab replay` matches.

Score, verify, metrics
----------------------
`score()` -> `expected_cum` (sum of W), `actual_cum` (sum of R), `W_last`, `W_by_round`,
`R_by_round`, `allocation` (site -> current count), `unassigned`, `assignment` (site -> p) and
`references` (`all_at_best`: everyone at the best site; `random`: E[W] with uniform independent
picks; `optimum`: the best allocation of all agents, with its W, found by the exact marginal-gain
greedy in `optimum_counts`: each next agent goes to the site with the largest gain
p_k * (1 - p_k) ** n_k, ties to the earlier site; exact because each site's gains decrease, and
O(K + N log K), so it stays instant for any agent and site count).
`verify()` -> `{"probabilities": assignment, "trial_key", "rounds"}`.

Metrics (entry points; all fold the `round_resolved` events, so `swarmlab replay` reproduces them;
denominator: agents holding a site, the sum of `counts`): `hidden.expected_W` (W of the current
round), `hidden.expected_cum`, `hidden.actual_points` (R of the current round), `hidden.actual_cum`.

`report_sections` adds one table to `swarmlab report`: per run, `expected_cum`, `actual_cum` and
the per-round W of the three references, which the generic score table skips (nested dict).

`render_state()` (the replay page's world panel) -> a table of sites with the hidden p, the
current count and whether the site scored in the round just resolved, plus that round's W and
points.

Snapshots carry game state only (assignment, trial key, round, positions, the choices of this
round, resolved rounds); constructor config is kept on restore, as in ColoringGrid. So a fork or
resume under an edited spec of the same world type keeps the new `rounds`, `show_history` and
`permute` (`permute` only matters at reset, so it has no effect after a restore; `rounds` must
still equal `max_rounds`). The site set cannot change: `restore` raises `ValueError` when the
restored assignment's sites differ from the configured `n_sites` labels, before any schema is
advertised or action accepted. Changed `probabilities` over the same labels are not detected and
have no effect: the hidden assignment comes from the snapshot.

Fake script
-----------
`fake_chooser` (entry point `hidden_chooser`, model `fake:hidden_chooser`) is a free, deterministic
plumbing check: agent aNNN takes site number NNN mod n_sites (12 agents on 8 sites:
2,2,2,2,1,1,1,1 over A..H), and in round 1 tries a second choice that must be rejected.

Decisions the contract leaves open
----------------------------------
- `terminal()` always False: the run length is `max_rounds`, which `requires` ties to `rounds`.
- The one-choice rule is enforced at commit, not in `validate`, so a rejected second choice is
  visible in the log as `accepted=False` (and in replay).
- Agents start unassigned and an unassigned agent makes no attempt; there is no "leave" action.
- `description()` falls back to twelve agents before `reset` (the count comes from `reset`).
"""
from __future__ import annotations

import heapq
import random
import re
from functools import lru_cache
from typing import Any, ClassVar

from ..ids import AgentId
from ..metrics.base import Metric
from ..rng import derive
from ..view import Observation, text_observation
from .base import Outcome, Requirements, World, tool

DEFAULT_PROBABILITIES = (0.60, 0.35, 0.25, 0.15, 0.10, 0.06, 0.03, 0.01)
N_AGENTS = 12   # the agent count `description()` states before `reset` knows the real one
N_ROUNDS = 20
ALREADY_CHOSE = "already_chose_this_round"
POINTS_PENDING = "not yet (points are scored after each full round)"
POSITION_PREFIX = "Your current position:"

ROUND_RE = re.compile(r"^Round (\d+) of (\d+)\.$")
AGENT_RE = re.compile(r"You are agent (a\d{3})\b")


# ---- pure helpers (shared by world, metrics, analysis) ----------------------------------------

def site_labels(n_sites: int) -> list[str]:
    return [chr(65 + i) for i in range(n_sites)]


_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
         "nineteen"]
_TENS = ["_", "_", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_word(n: int) -> str:
    """English words for 0..99 (e.g. 12 -> "twelve", 20 -> "twenty"); digits beyond."""
    if 0 <= n < 20:
        return _ONES[n]
    if 20 <= n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] + ("" if o == 0 else "-" + _ONES[o])
    return str(n)


def expected_points(counts: dict[str, int], probs: dict[str, float]) -> float:
    """W(n) = sum_k [1 - (1 - p_k)^n_k], summed in the order of `probs`."""
    return sum(1.0 - (1.0 - p) ** int(counts.get(k, 0)) for k, p in probs.items())


def trial_success(trial_key: int, round: int, agent: str, site: str, p: float) -> bool:
    """Outcome of `agent`'s attempt at `site` in `round` (a fixed, label-derived draw)."""
    return derive(trial_key, "trial", round, agent, site).random() < p


def resolve_round(round: int, positions: dict[str, str | None], probs: dict[str, float],
                  trial_key: int) -> dict[str, Any]:
    """Resolve one round from its end-of-round positions. Pure.

    Returns {"round", "counts" {site: n}, "outcomes" {agent: [site, success]} (assigned agents
    only, in agent order), "R" (sites with >= 1 success), "W" (expected points)}."""
    counts = {k: 0 for k in probs}
    outcomes: dict[str, list] = {}
    scored: set[str] = set()
    for agent in sorted(positions):
        site = positions[agent]
        if site is None:
            continue
        counts[site] += 1
        ok = trial_success(trial_key, round, agent, site, probs[site])
        outcomes[agent] = [site, ok]
        if ok:
            scored.add(site)
    return {"round": round, "counts": counts, "outcomes": outcomes, "R": len(scored),
            "W": expected_points(counts, probs)}


def random_reference(n_agents: int, probs: list[float] | tuple[float, ...]) -> float:
    """E[W] when each agent picks one of the sites independently and uniformly at random
    (callers pass the probabilities sorted, so the float result does not depend on the labels)."""
    m = len(probs)
    return sum(1.0 - (1.0 - p / m) ** n_agents for p in probs)


@lru_cache(maxsize=32)
def optimum_counts(n_agents: int, probs: tuple[float, ...]) -> tuple[tuple[int, ...], float]:
    """The allocation of all n_agents to the sites (counts in the order of `probs`) that maximises
    W = sum_k 1 - (1 - p_k) ** n_k, and that W.

    Exact marginal-gain greedy: adding an agent to site k (holding n_k) gains p_k * (1 - p_k) ** n_k,
    a decreasing sequence in n_k, and W is a sum of per-site concave terms, so assigning each next
    agent to the site with the largest gain is optimal. A heap keeps it O(K + N log K). Equal gains
    go to the lower site index (the earlier position in `probs`)."""
    k = len(probs)
    counts = [0] * k
    heap = [(-p, i) for i, p in enumerate(probs)]
    heapq.heapify(heap)
    for _ in range(n_agents if k else 0):
        _, i = heapq.heappop(heap)
        counts[i] += 1
        p = probs[i]
        heapq.heappush(heap, (-(p * (1.0 - p) ** counts[i]), i))
    w = sum(1.0 - (1.0 - p) ** c for p, c in zip(probs, counts))
    return tuple(counts), w


def references(n_agents: int, assignment: dict[str, float]) -> dict[str, Any]:
    """all_at_best, random and optimum (allocation by site label, W*) for `assignment`."""
    sites = list(assignment)
    best_site = max(sites, key=lambda k: (assignment[k], -sites.index(k)))
    ordered = tuple(assignment[k] for k in sites)
    opt, w_opt = optimum_counts(n_agents, ordered)
    all_best = {k: (n_agents if k == best_site else 0) for k in sites}
    return {
        "all_at_best": {"allocation": all_best, "W": expected_points(all_best, assignment)},
        "random": {"W": random_reference(n_agents, sorted(ordered, reverse=True))},
        "optimum": {"allocation": dict(zip(sites, opt)), "W": w_opt},
    }


# ---- parse helpers ----------------------------------------------------------------------------

def parse_observation(text: str) -> dict[str, Any]:
    """Invert `HiddenSites.observe` (the last observation in `text` wins). Returns {round,
    rounds, position, previous, history, points_last, points_total}; {} when absent.
    `position` is None for "none"; `previous` is None or (site, success); `points_last` is None
    before the first scored round."""
    lines = text.split("\n")
    start = max((i for i, ln in enumerate(lines) if ROUND_RE.match(ln.strip())), default=-1)
    if start < 0:
        return {}
    out: dict[str, Any] = {}
    for raw in lines[start:]:
        line = raw.strip()
        if m := ROUND_RE.match(line):
            out["round"], out["rounds"] = int(m.group(1)), int(m.group(2))
        elif line.startswith(POSITION_PREFIX):
            v = line[len(POSITION_PREFIX):].strip()
            out["position"] = None if v == "none" else v
        elif line.startswith("Your previous attempt:"):
            v = line.split(":", 1)[1].strip()
            m2 = re.match(r"site ([A-Z]), (success|failure)$", v)
            out["previous"] = (m2.group(1), m2.group(2) == "success") if m2 else None
        elif line.startswith("Your attempts so far:"):
            out["history"] = line.split(":", 1)[1].strip()
        elif line.startswith("Team points last round:"):
            v = line.split(":", 1)[1].strip()
            out["points_last"] = int(v) if v.isdigit() else None
        elif line.startswith("Team points so far:"):
            out["points_total"] = int(line.split(":", 1)[1].strip())
    return out


def parse_agent_id(system_text: str) -> str | None:
    m = AGENT_RE.search(system_text)
    return m.group(1) if m else None


def _plural(n: int, word: str, plural: str | None = None) -> str:
    return f"{n} {word if n == 1 else (plural or word + 's')}"


# ---- world ------------------------------------------------------------------------------------

class HiddenSites(World):
    """Allocation under a strict information boundary: hidden fixed site qualities, renewable discoveries."""

    entry_point: ClassVar[str | None] = "hidden_sites"
    name = "hidden_sites"
    requires: ClassVar[Requirements] = Requirements(commit="immediate", max_rounds="rounds")

    def __init__(self, probabilities: list[float] | tuple[float, ...] = DEFAULT_PROBABILITIES,
                 n_sites: int = 8,
                 rounds: int = N_ROUNDS, show_history: bool = True, permute: bool = True) -> None:
        probs = [float(p) for p in probabilities]
        if not isinstance(n_sites, int) or not 1 <= n_sites <= 26:
            raise ValueError(f"n_sites must be an integer in 1..26, got {n_sites!r}")
        if len(probs) != n_sites:
            raise ValueError(f"probabilities has {len(probs)} values for {n_sites} sites")
        if not all(0.0 < p < 1.0 for p in probs):
            raise ValueError("probabilities must be in (0, 1)")
        if not isinstance(rounds, int) or rounds < 1:
            raise ValueError(f"rounds must be a positive integer, got {rounds!r}")
        self.probabilities = probs
        self.n_sites = n_sites
        self.rounds = rounds
        self.show_history = bool(show_history)
        self.permute = bool(permute)
        self.sites = site_labels(n_sites)
        # ---- state (snapshotted) ----
        self.agents: list[str] = []
        self.assignment: dict[str, float] = {}      # hidden truth: site -> p
        self.trial_key = 0
        self.round = 0
        self.site_of: dict[str, str | None] = {}
        self.chose_this_round: set[str] = set()
        self.allocations: list[dict[str, str | None]] = []   # end-of-round positions, rounds 1..
        self.attempts: dict[str, list[list]] = {}           # agent -> [[round, site, success]]
        self.points_by_round: list[int] = []                 # R_r of resolved rounds
        self.expected_by_round: list[float] = []             # W_r of resolved rounds

    _skip_in_snapshot: ClassVar[tuple[str, ...]] = (
        "params", "probabilities", "n_sites", "rounds", "show_history", "permute", "sites")

    def restore(self, blob: bytes) -> None:
        """Restore game state, then refuse a site set that differs from the configured one."""
        super().restore(blob)
        restored = list(self.assignment)
        if set(restored) != set(self.sites) or len(restored) != self.n_sites:
            raise ValueError(
                f"hidden_sites: the snapshot has sites {', '.join(restored) or 'none'} but this "
                f"world is configured with sites {', '.join(self.sites)}; n_sites/probabilities "
                "(the site set) cannot change across a fork or resume (rounds, show_history and "
                "permute can)")

    # ---- lifecycle ---------------------------------------------------------------------------
    def reset(self, rng: random.Random, agents: list[AgentId]) -> None:
        probs = list(self.probabilities)
        if self.permute:
            rng.shuffle(probs)
        self.assignment = dict(zip(self.sites, probs))
        self.trial_key = rng.getrandbits(64)
        self.agents = [str(a) for a in agents]
        self.round = 0
        self.site_of = {a: None for a in self.agents}
        self.chose_this_round = set()
        self.allocations = []
        self.attempts = {a: [] for a in self.agents}
        self.points_by_round = []
        self.expected_by_round = []

    def begin_round(self, round: int) -> None:
        self.round = round
        self.chose_this_round = set()

    def end_round(self, round: int) -> dict:
        """Resolve round `round` from the positions after its last commit (logged as round_resolved)."""
        positions = dict(self.site_of)
        res = resolve_round(round, positions, self.assignment, self.trial_key)
        self.allocations.append(positions)
        for agent, (site, ok) in res["outcomes"].items():
            self.attempts.setdefault(agent, []).append([round, site, ok])
        self.points_by_round.append(res["R"])
        self.expected_by_round.append(res["W"])
        return {"W": res["W"], "R": res["R"], "counts": res["counts"], "outcomes": res["outcomes"]}

    def terminal(self) -> bool:
        return False  # max_rounds ends the run

    # ---- evaluator-side state helpers ----------------------------------------------------------
    def counts(self) -> dict[str, int]:
        occ = {k: 0 for k in self.sites}
        for s in self.site_of.values():
            if s is not None:
                occ[s] += 1
        return occ

    # ---- agent-facing --------------------------------------------------------------------------
    def _site_range(self) -> str:
        return self.sites[0] if len(self.sites) == 1 else f"{self.sites[0]} to {self.sites[-1]}"

    def description(self) -> str:
        n = len(self.agents) or N_AGENTS
        common = (
            f"Your team has {number_word(n)} agents and {number_word(self.rounds)} rounds to "
            "maximize total discovery points. Sites have different, unknown success "
            "probabilities that remain fixed throughout the experiment. Each site offers one "
            "discovery per round. Every agent there independently attempts to find it. More "
            "agents increase the chance of finding it, but the team receives only one point for "
            "that discovery, regardless of how many agents find it. Sites offer fresh "
            "discoveries each round. You can communicate through the shared message board and "
            "change sites."
        )
        operational = (
            f"There are {number_word(self.n_sites)} sites, named {self._site_range()}. All "
            "agents start unassigned, and an unassigned agent makes no attempt. You can move to a "
            "site with choose_site, at most once per round: after one accepted choice, another "
            "call in the same round is rejected. If you do not choose, you keep your current "
            "site. Choosing a site is not an attempt: at the end of each round, every "
            "assigned agent makes exactly one independent attempt at the site it holds then. "
            "After each round you are told privately whether your own attempt succeeded, and "
            "everyone is told the team's points for that round and the total so far. Nobody is "
            "told where the other agents are or how their attempts went, except through what "
            "they post on the message board."
        )
        return f"{common}\n\n{operational}"

    def _history_line(self, agent: str) -> str:
        per: dict[str, list[int]] = {}
        for _, site, ok in self.attempts.get(agent, []):
            n_ok = per.setdefault(site, [0, 0])
            n_ok[0] += 1
            n_ok[1] += int(ok)
        if not per:
            return "none"
        return "; ".join(f"{k}: {_plural(a, 'attempt')}, {_plural(s, 'success', 'successes')}"
                         for k, (a, s) in per.items())

    def observe(self, agent: AgentId) -> Observation:
        r = max(self.round, 1)  # `swarmlab prompts` renders after begin_round(1)
        mine = self.site_of.get(agent)
        last = next((x for x in reversed(self.attempts.get(agent, [])) if x[0] == r - 1), None)
        prev = "none yet" if last is None else f"site {last[1]}, {'success' if last[2] else 'failure'}"
        resolved = self.points_by_round[: r - 1]
        lines = [f"Round {r} of {self.rounds}.", f"{POSITION_PREFIX} {mine or 'none'}",
                 f"Your previous attempt: {prev}"]
        if self.show_history:
            lines.append(f"Your attempts so far: {self._history_line(agent)}")
        lines.append(f"Team points last round: {resolved[-1] if resolved else POINTS_PENDING}")
        lines.append(f"Team points so far: {sum(resolved)}")
        return text_observation("\n".join(lines))

    @tool("choose_site", lambda self: f"Hold site {self._site_range()} from now on, or keep your "
          "current one. At most one accepted choice per round; if you do not choose, you keep your site.",
          {"site": lambda self: {"type": "string", "enum": list(self.sites)}})
    def choose_site(self, agent: AgentId, site: str) -> Outcome:
        if site not in self.sites:
            return Outcome(accepted=False, feedback={
                "error": "unknown_site", "detail": f"site must be one of {', '.join(self.sites)}"})
        if agent not in self.site_of:
            return Outcome(accepted=False, feedback={"error": "unknown agent"})
        if agent in self.chose_this_round:
            return Outcome(accepted=False, feedback={"error": ALREADY_CHOSE})
        prev = self.site_of[agent]
        self.site_of[agent] = site
        self.chose_this_round.add(agent)
        return Outcome(accepted=True, feedback={"position": site, "previous": prev})

    # ---- evaluator-only ------------------------------------------------------------------------
    def score(self) -> dict:
        w, pts = list(self.expected_by_round), list(self.points_by_round)
        occ = self.counts()
        return {
            "expected_cum": sum(w), "actual_cum": sum(pts),
            "W_last": w[-1] if w else None, "W_by_round": w, "R_by_round": pts,
            "allocation": occ, "unassigned": sum(1 for s in self.site_of.values() if s is None),
            "assignment": dict(self.assignment),
            "references": references(len(self.agents) or N_AGENTS, self.assignment)
            if self.assignment else {},
        }

    def verify(self) -> dict:
        return {"probabilities": dict(self.assignment), "trial_key": self.trial_key,
                "rounds": self.rounds}

    def render_state(self) -> dict | None:
        occ = self.counts()
        r = len(self.points_by_round)
        scored = {site for xs in self.attempts.values() for rr, site, ok in xs if rr == r and ok}
        return {"sites": [{"site": k, "hidden p": p, "agents": occ[k], "scored this round": k in scored}
                          for k, p in self.assignment.items()],
                "W this round": self.expected_by_round[-1] if r else None,
                "points this round": self.points_by_round[-1] if r else None}

    def report_sections(self, arms: dict[str, list[Any]]) -> list[str]:
        from ..report import f, table

        rows = []
        for arm, runs in arms.items():
            for run in runs:
                sc = run.score if isinstance(run.score, dict) else {}
                ref = sc.get("references") or {}
                rows.append([arm, run.id, len(sc.get("W_by_round") or []), f(sc.get("expected_cum")),
                             sc.get("actual_cum", "n/a"),
                             *(f((ref.get(k) or {}).get("W")) for k in ("all_at_best", "random", "optimum"))])
        head = ["arm", "run", "rounds", "expected_cum", "actual_cum", "all_at_best W", "random W",
                "optimum W"]
        return ["## Hidden Sites: points and references", "",
                ("References are expected points per round; multiply by rounds to compare with "
                 "expected_cum."), "", *table(head, rows), ""]


# ---- metrics ----------------------------------------------------------------------------------

class _HiddenFold(Metric):
    """Folds the world's `round_resolved` events (W, R and the site counts of each round)."""

    entry_point: ClassVar[str | None] = None

    def __init__(self) -> None:
        self.W: list[float] = []
        self.R: list[int] = []
        self.assigned = 0

    def update(self, event: Any) -> None:
        if getattr(event, "type", None) == "round_resolved":
            self.W.append(float(event.data["W"]))
            self.R.append(int(event.data["R"]))
            self.assigned = sum(event.data["counts"].values())

    def _pick(self) -> float:
        raise NotImplementedError

    def value(self) -> tuple[float | None, int]:
        return (self._pick(), self.assigned) if self.W else (None, 0)


class ExpectedW(_HiddenFold):
    entry_point: ClassVar[str | None] = "hidden.expected_W"
    name = "hidden.expected_W"
    description = "expected points W of the end-of-round allocation (denominator: assigned agents)"

    def _pick(self):
        return self.W[-1]


class ExpectedCum(_HiddenFold):
    entry_point: ClassVar[str | None] = "hidden.expected_cum"
    name = "hidden.expected_cum"
    description = "cumulative expected points, sum of W over rounds so far"

    def _pick(self):
        return sum(self.W)


class ActualPoints(_HiddenFold):
    entry_point: ClassVar[str | None] = "hidden.actual_points"
    name = "hidden.actual_points"
    description = "team points this round (sites with at least one success)"

    def _pick(self):
        return float(self.R[-1])


class ActualCum(_HiddenFold):
    entry_point: ClassVar[str | None] = "hidden.actual_cum"
    name = "hidden.actual_cum"
    description = "cumulative team points so far"

    def _pick(self):
        return float(sum(self.R))


# ---- fake LLM script (entry point `hidden_chooser`, model "fake:hidden_chooser") ----------------

def fake_site(agent: str, sites: list[str] | None = None) -> str:
    """Site number (agent index mod len(sites)) of `sites` (default A..H): a000 -> A, ...,
    a007 -> H, a008 -> A, ..., a011 -> D, i.e. allocation (2,2,2,2,1,1,1,1) on eight sites."""
    sites = sites or site_labels(8)
    return sites[int(agent[1:]) % len(sites)]


def fake_chooser(request: Any, rng: random.Random) -> Any:
    """FakeProvider script for the plumbing dry run.

    Finds its agent id in the system prompt, the site labels in the `choose_site` schema, and
    the round and position in the latest observation. One response per turn:
    - round 1: choose_site(fake_site(agent, sites)), post("a003: at D"), choose_site(same) a
      second time (must be rejected: one choice per turn), end_turn;
    - later rounds: read_board, choose_site(current position) (keep), end_turn.
    Expected with 12 agents on 8 sites: allocation (2,2,2,2,1,1,1,1) over A..H every round and
    12 rejected second choices in round 1 only. A probe (no tools) gets the text
    `{"site": <site>}`."""
    import json

    from ..providers.base import text_of
    from ..providers.fake import RESULTS_PREFIX, _assistant_calls, _response

    tools = {t.name for t in request.tools}
    choose = next((t for t in request.tools if t.name == "choose_site"), None)
    sites = (choose.parameters.get("properties", {}).get("site", {}).get("enum")
             if choose is not None else None) or None
    msgs = request.messages
    agent = next((a for m in msgs if m.role == "system" and (a := parse_agent_id(text_of(m.content)))),
                 None)
    obs = next((text_of(m.content) for m in reversed(msgs)
                if m.role == "user" and POSITION_PREFIX in text_of(m.content)), "")
    parsed = parse_observation(obs)
    rnd = parsed.get("round", 1)
    site = parsed.get("position") or (fake_site(agent, sites) if agent else (sites or ["A"])[0])
    if not tools:
        return _response(request, rng, [], json.dumps({"site": site}))
    last_user = max((i for i, m in enumerate(msgs) if m.role == "user"
                     and not text_of(m.content).startswith(RESULTS_PREFIX)), default=-1)
    this_turn = [name for m in msgs[last_user + 1:] for name, _ in _assistant_calls(m)]
    calls: list[tuple[str, dict]] = []
    if not this_turn:
        if rnd > 1 and "read_board" in tools:
            calls.append(("read_board", {}))
        if "choose_site" in tools:
            calls.append(("choose_site", {"site": site}))
        if rnd == 1 and "post" in tools:
            calls.append(("post", {"text": f"{agent}: at {site}"}))
        if rnd == 1 and "choose_site" in tools:
            calls.append(("choose_site", {"site": site}))
    if "end_turn" in tools:
        calls.append(("end_turn", {}))
    return _response(request, rng, calls)
