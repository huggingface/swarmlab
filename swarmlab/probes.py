"""Probes: out-of-band questions answered by the agent's own model (docs/INTERFACE-M1b.md §5).

```python
class Probe(Plugin):
    name: str; every: int = 1
    def question(self, agent, round) -> str        # Jinja2 template rendered with agent, round
    def parse(self, text, candidates=None) -> tuple[bool, dict]   # (ok, parsed)
    def coder_model(self) -> str | None            # cheap model for free-text parsing; None = local only
    def candidates_from_context(self, context) -> list[str] | None   # known answer names, if any
class BeliefProbe(Probe):   entry_point = "belief"
```

Candidates hook: the runner calls `probe.candidates_from_context(participant.probe_context())`
once per agent and passes the result to every `parse` of that probe answer (also the coder
model's reply). The base returns None. `BeliefProbe` reads the names from the latest FlagGame
observation in the context (`flaggame.parse_observation` on a text part that starts with the
"Candidate flags:" preamble; in the image modality the names listed in that part's framing line,
`flaggame.candidate_names`); any other world yields None and parsing stays name-agnostic.

Tolerant parsing (`BeliefProbe.parse`, after the 2026-10-06 smoke where 9 of 12 Qwen3.5-9B answers
did not parse, mostly because reasoning used the whole token budget):

- reasoning is removed first (`strip_reasoning`: closed `<think>...</think>` blocks, a dangling
  `</think>` prefix, an unterminated `<think>` tail);
- the first JSON object anywhere in the text is used (prose and code fences around it are fine);
  if there is none, a `"candidate": "<name>"` pair inside truncated JSON is still accepted and
  marked `parsed["partial"] = True`;
- the candidate key may be `candidate`, `answer`, `guess` or `flag` (keys matched
  case-insensitively, first match in that order);
- with known candidate names, the value is matched case-insensitively, also after stripping a
  leading "candidate"/"flag" word and surrounding punctuation ("candidate c" -> "C"); a value that
  matches no name keeps its text and is marked `parsed["unknown_candidate"] = True` (still ok: the
  agent answered, it is just wrong).

Runner hook (swarmlab/runner.py `_probe_round`): in round r, after the world and board commit
(`action_committed*`) and before the round's `metric*` events, for every probe with
`r % every == 0` and every live agent, in the round's seeded order:

- a participant without `probe_context()` (scripted) gets one `probe` event with `ok=False,
  parsed={"skipped": "no_context"}` per run, per probe, per agent (re-derived from the log on
  resume and fork, so it is never repeated);
- otherwise the request is `ChatRequest(messages=probe_messages(context) + [user(question)],
  tools=[], **participant.model_request_defaults())`, sent with `executor.infer(agent, req,
  "measurement")`; agents are probed concurrently and their events are logged in seeded order;
- the answer is parsed locally; if that fails and `coder_model()` is set, one extraction call on
  the coder model (`CODER_SYSTEM` + question + answer) is made, also under `measurement`, and its
  reply is parsed with the same `parse` (`parsed["coded"] = True` marks such answers);
- the event is `probe{probe, question_hash, raw_hash, parsed, ok, cost_usd}`: `question_hash` and
  `raw_hash` are blob shas of the question text and the raw answer text (both stored), `cost_usd`
  is the *nominal* cost of the probe's calls (as first produced, also on a cache hit) so a resumed
  run's logical view equals an uninterrupted one; actual spend is in the ledger.

Budget: `MeasurementBudgetReached` -> `ok=False, parsed={"skipped": "measurement_budget"}` for that
agent, and probing stops for the rest of this runner's life (a resume with a larger budget probes
again). `HardCeilingReached` during a probe -> `ok=False, parsed={"skipped": "hard_ceiling"}`; the
round is still committed (its turns are complete and its world commit already happened), then
the run ends with `run_ended(hard_ceiling_probes)` at round r (round r is kept; plain
`hard_ceiling` is the mid-round abort that discards the round). Resume continues from r + 1; the
skipped probes of round r are not retried.

Context: `probe_messages` flattens tool traffic into text (an assistant tool call becomes a text
line `[tool call] name {args}`, a tool message becomes a user text `[tool result] ...`) because
the probe request carries `tools=[]` and the Anthropic API rejects tool_use/tool_result blocks
when no tools are defined. The participant's own messages are never modified.
"""
from __future__ import annotations

import json
import re
from typing import Any, ClassVar

import jinja2

from .base import Plugin
from .providers.base import ChatMessage, text_of
from .view import Part

BELIEF_QUESTION = (
    "Which candidate do you currently believe the flag is? Answer with JSON "
    '{"candidate": "<name>", "confidence": <0..1>} and nothing else.'
)

CODER_SYSTEM = (
    "You extract structured answers. You are given a question that asked for a JSON answer and "
    "the free-text reply someone gave. Reply with only the JSON object the question asked for, "
    "using the reply's content; use null for a value the reply does not state."
)

_ENV = jinja2.Environment(undefined=jinja2.StrictUndefined, autoescape=False)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_CANDIDATE_KEYS = ("candidate", "answer", "guess", "flag")
_PAIR_RE = re.compile(r'"(candidate|answer|guess|flag)"\s*:\s*"([^"\n]{1,64})"', re.IGNORECASE)
_PREFIX_RE = re.compile(r"^(?:candidate|flag)\s*[:#]?\s*", re.IGNORECASE)


def strip_reasoning(text: str | None) -> str:
    """`text` without `<think>...</think>` blocks, a dangling `...</think>` prefix (the opening
    tag was in the prompt template) or an unterminated `<think>...` tail."""
    body = _THINK_RE.sub("", text or "")
    if "</think>" in body:
        body = body.rsplit("</think>", 1)[1]
    return re.sub(r"<think>.*\Z", "", body, flags=re.DOTALL)


def match_candidate(value: str, candidates: list[str] | None) -> str | None:
    """The known candidate `value` names (case-insensitive, prefix/punctuation tolerant), or None."""
    if not candidates:
        return None
    by_lower = {c.lower(): c for c in candidates}
    v = value.strip().strip("\"'`*.,;:!()[]{}<> ").strip()
    for attempt in (v, _PREFIX_RE.sub("", v).strip("\"'`*.,;:!()[]{}<> ")):
        if attempt.lower() in by_lower:
            return by_lower[attempt.lower()]
    return None


def parse_json_object(text: str) -> dict | None:
    """The first JSON object in `text` (code fences and <think> blocks tolerated), or None."""
    body = _THINK_RE.sub("", text or "")
    start = body.find("{")
    while start >= 0:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(body)):
            ch = body[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(body[start:i + 1])
                    except ValueError:
                        break
                    if isinstance(value, dict):
                        return value
                    break
        start = body.find("{", start + 1)
    return None


class Probe(Plugin):
    """Base class. Subclasses set `name`/`template` and implement `parse`."""

    name: str = "probe"
    every: int = 1
    template: str = ""

    def question(self, agent: Any, round: int) -> str:
        return _ENV.from_string(self.template).render(agent=str(agent), round=round)

    def parse(self, text: str, candidates: list[str] | None = None) -> tuple[bool, dict]:
        raise NotImplementedError

    def coder_model(self) -> str | None:
        return getattr(self, "_coder_model", None)

    def candidates_from_context(self, context: list[ChatMessage]) -> list[str] | None:
        """Known answer names for this agent (passed to `parse`), or None when unknown."""
        return None


class BeliefProbe(Probe):
    """Ask for the agent's current belief; ok requires a string `candidate`."""

    entry_point: ClassVar[str | None] = "belief"

    def __init__(self, every: int = 1, coder_model: str | None = None, name: str = "belief",
                 question: str = BELIEF_QUESTION) -> None:
        if every < 1:
            raise ValueError("every must be >= 1")
        self.every = every
        self.name = name
        self.template = question
        self._coder_model = coder_model

    def candidates_from_context(self, context: list[ChatMessage]) -> list[str] | None:
        from .world.flaggame import PREAMBLE, candidate_names

        for m in reversed(context):
            if m.role != "user" or isinstance(m.content, str):
                continue
            for part in m.content:
                if part.type == "text" and (part.text or "").lstrip().startswith(PREAMBLE):
                    names = candidate_names(part.text or "")
                    if names:
                        return names
        return None

    def parse(self, text: str, candidates: list[str] | None = None) -> tuple[bool, dict]:
        body = strip_reasoning(text)
        obj = parse_json_object(body)
        parsed: dict[str, Any] = {}
        if obj is None:
            pair = _PAIR_RE.search(body)
            if pair is None:
                return False, {"error": "no JSON object"}
            obj = {pair.group(1).lower(): pair.group(2)}
            parsed["partial"] = True
        lower = {str(k).lower(): v for k, v in obj.items()}
        cand = next((lower[k] for k in _CANDIDATE_KEYS if k in lower), None)
        if isinstance(cand, (int, float)) and not isinstance(cand, bool):
            cand = str(cand)
        if not isinstance(cand, str) or not cand.strip():
            return False, {"error": "no candidate", "answer": obj}
        known = match_candidate(cand, candidates)
        parsed = {"candidate": known or cand.strip(), **parsed}
        if candidates and known is None:
            parsed["unknown_candidate"] = True
        conf = lower.get("confidence")
        if isinstance(conf, (int, float)) and not isinstance(conf, bool):
            parsed["confidence"] = float(conf)
        return True, parsed


def build_probe(spec: Any) -> Probe:
    """A `Probe` from an instance, an entry-point name, or a `{type, params}` mapping/PluginSpec."""
    import copy

    from .registry import build

    if isinstance(spec, Probe):
        return copy.deepcopy(spec)
    if isinstance(spec, str):
        spec = {"type": spec, "params": {}}
    probe = build(spec, "swarmlab.probes")
    if not isinstance(probe, Probe):
        raise TypeError(f"{spec!r} does not build a Probe")
    return probe


def probe_messages(context: list[ChatMessage]) -> list[ChatMessage]:
    """The agent's context with tool calls and tool results rendered as text (see module doc)."""
    out: list[ChatMessage] = []
    for m in context:
        if m.role == "tool":
            out.append(ChatMessage(role="user", content=f"[tool result] {text_of(m.content)}"))
        elif m.role == "assistant" and m.tool_calls:
            lines = [f"[tool call] {tc.name} {json.dumps(tc.args, sort_keys=True)}" for tc in m.tool_calls]
            text = text_of(m.content)
            out.append(ChatMessage(role="assistant", content="\n".join(([text] if text else []) + lines)))
        else:
            content = m.content if isinstance(m.content, str) else [Part(**p.model_dump()) for p in m.content]
            out.append(ChatMessage(role=m.role, content=content))
    return out
