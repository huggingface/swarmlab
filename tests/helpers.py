"""Importable test fixtures (plugins must live at module level so `module:Class` specs resolve)."""
from __future__ import annotations

import os
import time
from pathlib import Path

from swarmlab import (
    Board,
    Experiment,
    Outcome,
    Participant,
    TurnUsage,
    World,
    text_observation,
    tool,
)
from swarmlab.metrics.base import Metric
from swarmlab.participants import EvidenceAggregator
from swarmlab.world.flaggame import FlagGame

PAUSE_ENV = "SWARMLAB_TEST_PAUSE"
PAUSE_AT_ENV = "SWARMLAB_TEST_PAUSE_AT"  # which commit to block in (default 7)


class PausingFlagGame(FlagGame):
    """FlagGame that blocks inside its 7th commit when $SWARMLAB_TEST_PAUSE names a marker file.

    The 7th commit happens after round 7's turn events are in the log and before its
    `round_committed`, which gives the recovery test a deterministic kill window.
    """

    def commit(self, actions):
        self.commits = getattr(self, "commits", 0) + 1
        marker = os.environ.get(PAUSE_ENV)
        if marker and self.commits == int(os.environ.get(PAUSE_AT_ENV, "7")):
            Path(marker).write_text("paused")
            time.sleep(120)
        return super().commit(actions)


class Chatter(Participant):
    """Posts its agent id, reads the board, then yields without end_turn (no_tool)."""

    async def turn(self, view, tools):
        await tools.call("post", {"text": f"hello from {self.agent} r{view.round}"})
        res = await tools.call("read_board", {})
        self.seen = getattr(self, "seen", []) + [i["content"] for i in res.result["items"]]
        return TurnUsage(calls=2)


def flag_experiment(n_agents: int = 8, *, world=None, medium=None, name: str = "flag", **kw) -> Experiment:
    return Experiment(
        name=name,
        world=world if world is not None else FlagGame(),
        participants=[EvidenceAggregator()] * n_agents,
        medium=medium if medium is not None else Board(topology="gossip"),
        metrics=["belief.consensus", "belief.accuracy", "belief.entropy", "belief.polarization",
                 "comm.read_rate", "comm.posts_per_round", "comm.hops"],
        **kw,
    )


def logical(run, exclude=("seq", "ts")):
    return list(run_events(run, exclude))


def run_events(run, exclude=("seq", "ts")):
    from swarmlab.events import logical_view

    return logical_view(run.events_all, exclude=exclude)


class LouderAggregator(EvidenceAggregator):
    """Subclass of a registered participant; must serialise as itself, not as its parent."""

    def __init__(self, volume: int = 2) -> None:
        self.volume = volume


class BigFlag(FlagGame):
    """Subclass of the registered world with no constructor of its own."""


OPERATIONAL_HOOK: dict = {}


class Inferrer(Participant):
    """Calls OPERATIONAL_HOOK["fn"](agent, round) mid-turn (stands in for M1b's tools.infer)."""

    async def turn(self, view, tools):
        fn = OPERATIONAL_HOOK.get("fn")
        if fn is not None:
            fn(self.agent, view.round)
        await tools.call("post", {"text": f"{self.agent} r{view.round}"})
        await tools.call("end_turn", {})
        return TurnUsage(calls=2)


class PostThenRead(Participant):
    """The reviewer's A6 participant: post first, then read the board, every round."""

    async def turn(self, view, tools):
        await tools.call("post", {"text": f"{self.agent} r{view.round}"})
        await tools.call("read_board", {})
        return TurnUsage(calls=2)


class ReadThenPost(Participant):
    async def turn(self, view, tools):
        await tools.call("read_board", {})
        await tools.call("post", {"text": f"{self.agent} r{view.round}"})
        return TurnUsage(calls=2)


# ---- M1b fixtures: participants that infer through the harness ------------------------------------
TEST_PRICING = {"*": (10.0, 50.0, 1.0)}  # makes a 2-call fake reader turn cost about $0.01


class FakeLLM(Participant):
    """Minimal LLM loop over `tools.infer` (stands in for WP7's LLMAgent).

    Each turn starts a fresh conversation (system + one user message with the observation), so
    requests are a pure function of (agent, round, observation, board).
    """

    def __init__(self, model: str = "fake:flaggame_reader", max_tokens: int = 64,
                 max_calls: int = 4) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.max_calls = max_calls

    async def turn(self, view, tools):
        import json

        from swarmlab.providers.base import ChatMessage, ChatRequest

        text = "\n".join(p.text or "" for p in view.observation.parts if p.type == "text")
        msgs = [ChatMessage(role="system", content=f"You are agent {self.agent}."),
                ChatMessage(role="user", content=f"Round {view.round}.\n{text}")]
        calls = 0
        for _ in range(self.max_calls):
            resp = await tools.infer(ChatRequest(model=self.model, messages=msgs, tools=view.tools,
                                                 max_tokens=self.max_tokens))
            if not resp.tool_calls:
                break
            msgs.append(ChatMessage(role="assistant", content=resp.text, tool_calls=resp.tool_calls))
            ended = False
            for tc in resp.tool_calls:
                res = await tools.call(tc.name, tc.args)
                calls += 1
                msgs.append(ChatMessage(role="tool", content=json.dumps(res.model_dump(mode="json")),
                                        tool_call_id=tc.call_id))
                ended = ended or tc.name == "end_turn"
            if ended:
                break
        return TurnUsage(calls=calls, cost_usd=123.0)  # cost is overridden by the executor


class TwiceInferrer(Participant):
    """Sends the same request twice per turn (the second must hit the cache), then ends."""

    async def turn(self, view, tools):
        from swarmlab.providers.base import ChatMessage, ChatRequest

        req = ChatRequest(model="fake:flaggame_reader", max_tokens=32,
                          messages=[ChatMessage(role="user", content=f"hello {self.agent}")])
        first = await tools.infer(req)
        second = await tools.infer(req)
        self.seen = [first.cached, second.cached, second.cost_usd]
        await tools.call("end_turn", {})
        return TurnUsage(calls=1)


def llm_experiment(n_agents: int = 4, *, world=None, name: str = "llm", pricing=None,
                   provider=None, **kw) -> Experiment:
    from swarmlab.providers.fake import FakeProvider

    return Experiment(
        name=name,
        world=world if world is not None else FlagGame(),
        participants=[FakeLLM()] * n_agents,
        medium=Board(topology="broadcast"),
        metrics=["belief.consensus", "belief.accuracy"],
        providers={"fake": provider or FakeProvider(pricing=pricing or TEST_PRICING)},
        **kw,
    )


# ---- WP7 fixtures: LLMAgent experiments and scripted fake-provider behaviours -----------------------
def llm_agent_experiment(n_agents: int = 4, *, world=None, medium=None, name: str = "llmagent",
                         pricing=None, agent_kw=None, probes=(), metrics=None, **kw) -> Experiment:
    """`n_agents` LLMAgent(model="fake:reader") on the Flag Game over a fake provider."""
    from swarmlab.participants import LLMAgent
    from swarmlab.providers.fake import FakeProvider

    agent_kw = {"model": "fake:reader", **(agent_kw or {})}
    return Experiment(
        name=name,
        world=world if world is not None else FlagGame(),
        participants=[LLMAgent(**agent_kw)] * n_agents,
        medium=medium if medium is not None else Board(topology="broadcast"),
        metrics=metrics if metrics is not None else ["belief.consensus", "belief.accuracy"],
        probes=list(probes),
        providers={"fake": FakeProvider(pricing=pricing or TEST_PRICING)},
        **kw,
    )


def _resp(request, text: str = "", calls=(), finish: str | None = None):
    from swarmlab.providers.base import ChatResponse, Usage, model_id
    from swarmlab.tools import ToolCall

    tool_calls = [ToolCall(call_id=f"t{i}", name=n, args=a) for i, (n, a) in enumerate(calls)]
    return ChatResponse(text=text, tool_calls=tool_calls, usage=Usage(), cost_usd=0.0, provider="fake",
                        model=model_id(request), latency_s=0.0,
                        finish_reason=finish or ("tool_use" if tool_calls else "end_turn"))


def _last_text(request) -> str:
    from swarmlab.providers.base import text_of

    return text_of(request.messages[-1].content) if request.messages else ""


def script_bad_json_then_ok(request, rng):
    """json protocol: malformed array first; after the parse-error feedback, end the turn."""
    if "could not parse" in _last_text(request):
        return _resp(request, '[{"name": "end_turn", "args": {}}]')
    return _resp(request, 'Sure: [{"name": "guess", "args": {"candidate": ')


def script_always_bad_json(request, rng):
    return _resp(request, '[{"name": broken')


def script_end_turn_first(request, rng):
    """End the turn, then (same response) try to guess: the guess must be refused."""
    return _resp(request, calls=[("end_turn", {}), ("guess", {"candidate": "A"})])


def script_status_forever(request, rng):
    return _resp(request, calls=[("my_status", {})])


def script_raw_args(request, rng):
    """Native: first a call with unparseable arguments; after the error result, end the turn."""
    if request.messages and request.messages[-1].role == "tool":
        return _resp(request, calls=[("end_turn", {})])
    return _resp(request, calls=[("guess", {"_raw": "{candidate: A"})], finish="bad_tool_args")


def script_freetext_belief(request, rng):
    """The reader, but probe answers come as free text (not JSON)."""
    import json as _json

    from swarmlab.providers.fake import flaggame_reader

    resp = flaggame_reader(request, rng)
    if request.tools:
        return resp
    cand = _json.loads(resp.text).get("candidate")
    return _resp(request, f"I would say the flag is candidate {cand}, fairly sure.")


def script_coder(request, rng):
    """Extraction model: pull `candidate X` out of the reply in the user message."""
    import re

    m = re.search(r"candidate ([A-Z])", _last_text(request).split("Reply:")[-1])
    import json as _json

    return _resp(request, _json.dumps({"candidate": m.group(1), "confidence": None}) if m else "{}")


PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


class ImageFlagGame(FlagGame):
    """FlagGame whose observation also carries an image part (for image passthrough tests)."""

    def observe(self, agent):
        from swarmlab.view import Part

        obs = super().observe(agent)
        return obs.model_copy(update={"parts": [*obs.parts, Part(type="image", image_png_b64=PNG_B64)]})


# ---- field-notes fixtures: a tiny custom world (sites), its fake script and metric ---------------


class SiteWorld(World):
    """Agents pick one of four sites per round; the score counts picks per site."""

    SITES = ("A", "B", "C", "D")

    def reset(self, rng, agents):
        self.picks = {s: 0 for s in self.SITES}
        self.last: dict[str, str] = {}
        self.round = 0

    def begin_round(self, round):
        self.round = round

    def observe(self, agent):
        return text_observation(f"Sweep {self.round}. Sites: {' '.join(self.SITES)}. "
                                f"Picks so far: {self.picks}")

    @tool("choose_site", "Pick a site for this round.", {"site": "string"})
    def choose_site(self, agent, site: str) -> Outcome:
        if site not in self.picks:
            return Outcome(accepted=False, feedback={"error": "unknown_site"})
        self.picks[site] += 1
        self.last[str(agent)] = site
        return Outcome(accepted=True, feedback={"site": site, "crowd": self.picks[site]})

    def score(self):
        return dict(self.picks)

    def render_state(self):
        return {"picks": dict(self.picks), "round": self.round,
                "last_choice": [{"agent": a, "site": s} for a, s in sorted(self.last.items())]}


def site_script(request, rng):
    """fake:tests.helpers:site_script: pick a site (once per turn), then end_turn."""
    from swarmlab.providers.fake import _assistant_calls

    names = {t.name for t in request.tools}
    if not names:
        return _resp(request, '{"candidate": "A"}')
    last_user = max((i for i, m in enumerate(request.messages) if m.role == "user"), default=-1)
    done = [n for m in request.messages[last_user + 1:] for n, _ in _assistant_calls(m)]
    if "choose_site" in names and "choose_site" not in done:
        site = "ABCDX"[rng.randrange(5)]  # X: refused by the world
        return _resp(request, calls=[("choose_site", {"site": site})], finish="tool_use")
    return _resp(request, calls=[("end_turn", {})])


class CrowdMetric(Metric):
    """sites.max_share: the most-picked site's share of all picks so far."""

    entry_point = None
    name = "sites.max_share"

    def __init__(self) -> None:
        self.picks: dict[str, int] = {}

    def update(self, event):
        if getattr(event, "type", None) == "action_committed" and event.accepted:
            site = event.feedback.get("site")
            self.picks[site] = self.picks.get(site, 0) + 1

    def value(self):
        n = sum(self.picks.values())
        return (max(self.picks.values()) / n if n else None), n


class Crasher(Participant):
    """A participant whose turn raises (an errored turn)."""

    async def turn(self, view, tools):
        raise RuntimeError(f"provider said no in round {view.round}\nsecond line")


def site_experiment(name: str = "sites", n_llm: int = 3, crash: bool = True) -> Experiment:
    from swarmlab.participants import LLMAgent
    from swarmlab.providers.fake import FakeProvider

    parts = [LLMAgent(model="fake:tests.helpers:site_script")] * n_llm
    if crash:
        parts.append(Crasher())
    return Experiment(name=name, world=SiteWorld(), participants=parts,
                      medium=Board(topology="broadcast"), metrics=[CrowdMetric()],
                      providers={"fake": FakeProvider(pricing=TEST_PRICING)})
