"""Probes: out-of-band questions answered by the agent's own model (docs/INTERFACE-M1b.md §5).

```python
class Probe(Plugin):
    name: str; every: int = 1
    def question(self, agent, round) -> str        # Jinja2 template rendered with agent, round
    def parse(self, text) -> tuple[bool, dict]     # (ok, parsed)
    def coder_model(self) -> str | None            # cheap model for free-text parsing; None = local only
class BeliefProbe(Probe):   entry_point = "belief"
```

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
the run ends with `run_ended(hard_ceiling)` at round r. Resume continues from r + 1; the skipped
probes of round r are not retried.

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

    def parse(self, text: str) -> tuple[bool, dict]:
        raise NotImplementedError

    def coder_model(self) -> str | None:
        return getattr(self, "_coder_model", None)


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

    def parse(self, text: str) -> tuple[bool, dict]:
        obj = parse_json_object(text)
        if obj is None:
            return False, {"error": "no JSON object"}
        cand = obj.get("candidate")
        if not isinstance(cand, str) or not cand.strip():
            return False, {"error": "no candidate", "answer": obj}
        parsed: dict[str, Any] = {"candidate": cand.strip()}
        conf = obj.get("confidence")
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
