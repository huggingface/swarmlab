"""MiniSweAgent: mini-swe-agent 2.4.6, the official ProgramBench baseline agent, as a participant.

`MiniSweAgent(model, steps_per_turn=None, step_limit=1000, wall_limit_s=21600, max_tokens=8192,
temperature=None, extra=None, templates="solo", idle_until_assigned=False, wait_tool=False,
lead_channel="lead", worker_channels=None)`, entry point `mini_swe_agent`. No extra is needed:
mini-swe-agent's agent loop, exceptions and tool-call helpers are a vendored copy of 2.4.6 in
`swarmlab/vendor/minisweagent/` (MIT; its README lists the files, their wheel hashes and the
changes, which are import paths only).

The loop is mini-swe-agent's own `DefaultAgent` (`minisweagent/agents/default.py`, vendored): its
`query()` (step and wall limits checked before each model call, `n_model_calls`, cost),
`execute_actions()` and messages. swarmlab supplies the model and the environment through two
duck-typed adapters and drives `agent.step()` itself, because `run()` loops to the end of the
episode while a swarmlab turn may be only part of it. The words the model sees are the
baseline's: the ProgramBench templates (below), the `observation_template` and
`format_error_template` of the 2.4.6 `config/benchmarks/programbench.yaml` (copied into this
module; a test checks the copy against the real package when it is installed), rendered with
mini-swe-agent's template variables (`step_limit`, `n_model_calls`, `wall_time_limit_seconds`,
`elapsed_seconds`), so the step and time warnings appear exactly as in the baseline.

Two shapes
----------
- Solo (official): `steps_per_turn=None` (the default), one round, one turn (`commit: immediate`,
  `max_rounds: 1`, `options.max_calls_per_turn` above the number of commands the episode can
  run). The turn steps until the episode ends: a submission, the step limit (`LimitsExceeded`),
  the wall limit (`TimeExceeded`) or `max_consecutive_format_errors` (3) format errors in a row
  (`RepeatedFormatError`), each recorded as the baseline's `exit` message, or until the runner
  ends it (the executor's cap, the hard ceiling, cancellation). The `solo` template offers the
  model exactly the baseline's one tool, mini-swe-agent's `BASH_TOOL`: not the `end_turn` the
  executor always lists, so a model cannot end the episode early by calling it (it is an unknown
  tool, the baseline's format error, and the episode goes on).
- Swarm: `steps_per_turn=k` runs at most k steps (model calls) per turn; the episode continues at
  the agent's next turn from its stored messages. Typically a lead with the `lead` templates and
  workers with the `worker` templates and `idle_until_assigned` (docs/guide/programbench.md,
  "Swarm shape"). Under `commit: immediate` (lockstep rounds, the `fixed` scheduler) or
  `commit: async` (every agent loops over its own turns; k is then the tick cadence: each turn of
  k steps closes one tick). Under async the idle worker and `wait_for_workers` use the core
  `wait` tool (below), so the role of such an agent must list `wait` in its tools: the first turn
  raises a `ValueError` naming the fix when `view.tools` lacks it.

A finished agent (last message `exit`) never calls the model again: later turns return at once.

The turn
--------
1. Finished: return (note `finished`).
2. Deliveries: the board items new to this agent (by `(eligible_round, delivery_id)`, newer than
   the newest item inserted so far, so a push board without `push_consume` that re-pushes unread
   items is not inserted twice). Under push they come from `view.pushed`. Under pull nothing is
   read for the model (it reads with its `read_board` tool if it chooses), except by an idle
   worker (5).
3. First turn: build the `DefaultAgent` (its wall clock starts now, as the baseline's starts at
   construction), then do what `run()` does before its loop: `extra_template_vars |= {"task":
   "", ...}` and the rendered system and instance messages. Later turns rebuild the agent from
   the stored state (below).
4. New deliveries become one user message, `Delivered to you:` and a line per item `- <post_id>
   (round <r>): <content>` (`llm.render_delivered`, the `llm` participant's rendering; under
   `commit: async` without the ` (round <r>)`, as there).
5. `idle_until_assigned`: a worker without a current assignment returns before 3 with no model
   call (note `idle`) unless something new was delivered: pushed items, or else (when it has
   `read_board`) the items of one `read_board` call the participant makes itself. Something new
   is its assignment: it runs its steps. Its assignment ends when it posts to `lead_channel` (its
   report; the turn ends after that step) and it is idle again until the next delivery. The
   conversation is kept across assignments, as a resumed worker in the paper. Under `commit:
   async` (`view.turn` set) the participant calls the core `wait` instead of `read_board`, still
   with no model call: the turn parks until a post reaches the worker, and the posts `wait`
   returns (marked read) are the assignment, inserted as in 4. When `wait` returns `timed_out`
   with no posts (the deadlock rule: every other running agent is waiting or finished, so nobody
   can post), the turn ends with the note `idle`; the runner then starts the worker's next turn,
   which waits again. Such idle turns cost no model call but each closes a tick, so a worker
   that is never assigned work ends by its per-agent cap (`options.rounds_per_agent`), or when
   the run ends.
6. Steps in a worker thread (the bridge, below): up to `steps_per_turn`, handling
   `InterruptAgentFlow` exactly as `run()`: a `FormatError` adds its cost and messages (and the
   `RepeatedFormatError` exit at the limit), any other (`Submitted`, `LimitsExceeded`,
   `TimeExceeded`) appends its exit message. With the other templates the turn also ends after
   a step in which the model called `end_turn`, `wait_for_workers`, or a call was refused
   `turn_ended`; with `solo` none of these is offered (under `commit: async`
   `wait_for_workers` does not end the turn).

Model adapter
-------------
`query(messages)` turns mini-swe-agent's messages (system, user, assistant with OpenAI-style
`tool_calls`, tool; `extra` stripped as `_prepare_messages_for_api` does; `exit` skipped) into a
`ChatRequest` with the tools of `offered_tools(view)`, and awaits `tools.infer`. The `solo`
template offers `OFFICIAL_BASH` alone: mini-swe-agent 2.4.6's `BASH_TOOL` (from the vendored
`actions_toolcall.py`; its name, description and parameters unchanged, so no
`additionalProperties`; a test compares it with the real package when installed), and nothing
when the view has no `bash`. The other templates offer the executor's tools for this agent
(`view.tools`: `bash` and `end_turn`; with a board also `post` and `read_board`) plus
`wait_for_workers` when `wait_tool` (which therefore needs a template other than `solo`).
Provider adapters send a schema exactly as offered, so the wire schema is the baseline's (no
`additionalProperties`, and on Anthropic no `strict`). The reply becomes the assistant message
dict (`role`, `content`, `tool_calls` with `id`/`type`/`function.name`/`function.arguments`, and
the provider's `provider_content` when it carried thinking blocks, sent back unchanged) with
`extra = {"actions", "response", "cost"}` as `LitellmModel.query` returns it; `cost` is swarmlab's
`ChatResponse.cost_usd` and `response` a small summary (model, provider, finish reason, usage,
cached), since the full response is in the run's blobs. Timestamps are left out of `extra` so the
stored state is a function of the responses.

Tool calls are parsed with mini-swe-agent's own `parse_toolcall_actions`: `bash` calls are the
actions; a call to a tool that is not offered, bad `bash` arguments, or no tool call at all is
its `FormatError` (with the `format_error_template` and the finish reason in OpenAI's words:
`length`, `tool_calls`, `stop`), and nothing of that reply runs. Calls to other offered tools are
not actions: the adapter runs them itself (`post`, `read_board`, `end_turn` through `tools.call`;
`wait_for_workers` is answered without a call: it posts nothing and ends the turn after the
step) and records each result as a `tool` message, the JSON of `{"ok", "result", "error"}` (the
`llm` participant's format; `wait_for_workers` gets `waiting for workers; their reports arrive at
your next turn`). Under `commit: async` `wait_for_workers` is one call of the core `wait` (one
tool call against `max_calls_per_turn`) and does not end the turn: its tool message is the
posts it returned, rendered as a `read_board` result (`{"ok": true, "result": {"items": [...]},
"error": null}`), so the lead reads the reports inline; when `wait` timed out with no posts (no
worker can report) the message is `WAIT_STALLED`. Its description differs per mode
(`WAIT_SCHEMA` / `WAIT_SCHEMA_ASYNC`, chosen by `view.turn`). The core `wait` itself is never
offered to the model (it is the participant's own). All calls of a reply run in the order the
model issued them, and their tool messages follow in that order.

Environment adapter
-------------------
`execute(action)` runs `tools.call("bash", {"command": ...})` through the executor and returns the
world's feedback dict (`output`, `returncode`, `exception_info`), as the baseline's environment
returns its own. It raises `Submitted` (the baseline's exit message, submission = the rest of the
output) when the output submits (`world.programbench.submission_of`, the rule of `local.py`
`_check_finished`). A refused call (`ok=False`, or an action the world did not accept):

- `wall_limit` (the world's 6 h clock ran out): `TimeExceeded`, the baseline's exit message;
- `submitted` (another agent submitted): `Submitted` with an empty submission;
- anything else (`turn_ended`, `not_allowed`, ...): the observation `{"output": "",
  "returncode": -1, "exception_info": "swarmlab refused the command: <error>"}`, so the model
  sees it and the episode goes on.

`get_template_vars()` is `platform.uname()` of this machine (what mini-swe-agent's environments
give the `<system_information>` block; the world renders its `description()` the same way).

Bridge
------
`DefaultAgent.step()` is synchronous and calls the adapters synchronously, while `turn()` is a
coroutine and `tools.infer`/`tools.call` are coroutines on the runner's event loop. The steps run
in a worker thread (`asyncio.to_thread`); each adapter call schedules its coroutine on the loop
with `asyncio.run_coroutine_threadsafe(coro, loop).result()` and blocks the thread until it is
done. The executor's calls thus still run on the loop, one at a time, exactly as for any other
participant. An exception in a coroutine (`TurnCapReached`, `HardCeilingReached`, a provider
error) is re-raised in the thread, leaves `step()` (which does not catch it), and is re-raised by
`turn()` after the state is saved, so the runner ends the turn as usual (`cap`, `error`, abort).
Such an exception ends the turn, not the episode. Calls of the interrupted reply that got no
result get the baseline's "not executed" observation (`returncode -1`, `action was not
executed`), so the stored conversation never has a tool call without a result.

When the turn is cancelled (the runner cancels the other turns of a round on `HardCeilingReached`),
`turn()` cancels the bridge operation in flight (its task is tracked on the loop; the
`CancelledError` reaches `tools.infer`, whose gate releases the reservation and charges it if the
request was dispatched, as for any cancelled participant, or `tools.call`), refuses any further one
(the abort flag is checked under a lock before scheduling, and again when the operation starts),
and waits until the step thread has returned before it repairs the transcript as above, saves the
state and re-raises. So nothing is appended to the saved messages afterwards. The cancelled step's
model call is counted in `n_calls` (mini-swe-agent counts it before the call) and has no message. A
command in flight is killed by the world when its `bash` is cancelled (swarmlab/world/programbench.py,
Cancellation) and gets the "not executed" observation; a parked `wait` is released the same way.

Templates
---------
`templates` is `solo`, `lead`, `worker` (files in `prompts/minisweagent/`) or `file:<path>`, a
YAML mapping with `system_template` and `instance_template`, rendered by mini-swe-agent (Jinja2,
`StrictUndefined`) with its template variables plus: `programbench_system`,
`programbench_instance` (the official texts, `world.programbench.prompt_parts` with this machine's
uname), `programbench_rules` (the `<RULES>` block of the system text),
`programbench_command_rules` (the instance text from `## Command Execution Rules` on, without the
submission bullet), `resource_sentence` (the last paragraph of `view.description` when it starts
with `This machine has`, else empty), `lead_channel`, `worker_channels` (the `worker_channels`
param, or the channels of this agent's `post` tool other than `lead_channel`: for a lead one per
worker, for a worker its own), `wait_tool`, `turn_based` (`commit: async`: the lead's
`wait_for_workers` bullet says the tool returns the reports, the worker's text says it waits
between assignments; the round-based text is unchanged), `agent`, `role` and `tools` (names
offered). `solo`
is the official text plus the resource sentence, byte for byte what the baseline renders plus
that sentence; `lead` adds delegation instructions; `worker` has the environment rules and its
role, not the task statement. A role's `prompt_append` is added to the system message, its
`model` and `budget.max_tokens` replace the participant's.

State and snapshots
-------------------
Plain attributes only: `messages` (the agent's message dicts), `n_calls`, `cost`,
`n_consecutive_format_errors`, `elapsed_s` (the agent's clock), `finished`, `assigned`,
`delivered_upto`. The `DefaultAgent` and the adapters are never pickled: every turn rebuilds them
from this state, with `_start_time = time.time() - elapsed_s`, so the wall limit survives a resume
in a new process (time between processes is not counted). The clock runs only inside the agent's
own turns: in the solo shape that is the whole episode; in the swarm shape each agent counts its
own working time, and the world's clock still refuses commands after its wall limit
(`wall_limit` -> `TimeExceeded`). Under `commit: async` the clock is plain wall time within the
agent's turns, which includes the time parked in `wait` (`wait_for_workers` runs inside a step;
an idle worker's own wait is added to `elapsed_s`, also before its first assignment).

`turn()` returns an `LLMTurnUsage`: `calls` (tool calls made, the participant's own `read_board`
included), tokens and cost summed from the turn's responses (the runner overwrites them with the
executor's nominal totals), `finish_reasons` (so refusals are counted as for `llm`) and `notes`
(`idle`, `finished`, `exit:<status>` when the episode ended in this turn).

Params are in `params` (and the spec hash) only when they differ from the defaults; `model` is
always there.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import json
import platform
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import yaml

from ..providers.base import ChatMessage, ChatRequest, ChatResponse, parse_json_args
from ..tools import AgentTools, ToolCall, ToolSchema
from ..vendor.minisweagent import __version__ as msa_version
from ..vendor.minisweagent import exceptions as msa_exceptions
from ..vendor.minisweagent.actions_toolcall import (
    BASH_TOOL,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)
from ..vendor.minisweagent.default import DefaultAgent
from ..view import View
from ..world.programbench import SUBMIT_MARKER, prompt_parts, submission_of
from .base import Participant
from .llm import LLMTurnUsage, render_delivered

MSA_VERSION = "2.4.6"
TEMPLATES_DIR = Path(__file__).resolve().parent / "prompts" / "minisweagent"
TEMPLATE_SETS = ("solo", "lead", "worker")
WAIT_TOOL = "wait_for_workers"
# The solo template's only tool: mini-swe-agent 2.4.6's `BASH_TOOL` (vendored
# `actions_toolcall.py`, verbatim from the wheel), the one tool `LitellmModel` sends in the
# baseline, as a ToolSchema with its name, description and parameters unchanged.
OFFICIAL_BASH = ToolSchema(name=BASH_TOOL["function"]["name"],
                           description=BASH_TOOL["function"]["description"],
                           parameters=copy.deepcopy(BASH_TOOL["function"]["parameters"]))
WAIT_SCHEMA = ToolSchema(
    name=WAIT_TOOL,
    description="Pause until your workers report: your turn ends after this step, and their "
                "reports arrive by your next turn. Takes no arguments; posts nothing.",
    parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
)
WAIT_RESULT = "waiting for workers; their reports arrive at your next turn"
# commit: async: `wait_for_workers` is the core `wait` tool, which returns the reports
WAIT_SCHEMA_ASYNC = WAIT_SCHEMA.model_copy(update={
    "description": "Wait until your workers report: returns the messages posted to you since "
                   "you last read them (their reports), or says at once that no worker can "
                   "report. Takes no arguments; posts nothing."})
WAIT_STALLED = ("no worker can report: every worker is idle or finished, so nothing will arrive "
                "while you wait. Assign work or continue yourself.")
WAIT_NOT_LISTED = ("the mini_swe_agent participant with {flag} under commit: async waits with the "
                   "core `wait` tool, which agent {agent} is not offered: list `wait` in its "
                   "role's tools (roles: {{<role>: {{tools: [bash, post, read_board, wait]}}}})")
NOT_EXECUTED = {"output": "", "returncode": -1, "exception_info": "action was not executed"}
REFUSED_PREFIX = "swarmlab refused the command: "
# swarmlab finish reasons (Anthropic's stop reasons, OpenAI's) in OpenAI's words, as litellm
# hands them to mini-swe-agent's format_error_template
FINISH_REASONS = {"max_tokens": "length", "tool_use": "tool_calls", "end_turn": "stop",
                  "stop_sequence": "stop", "refusal": "content_filter"}

# ---- mini-swe-agent 2.4.6, config/benchmarks/programbench.yaml, `model:` (MIT; verbatim) ---------

OBSERVATION_TEMPLATE = """{% if output.exception_info -%}
<exception>{{output.exception_info}}</exception>
{% endif -%}
<returncode>{{output.returncode}}</returncode>
{% if output.output | length < 10000 -%}
<output>
{{ output.output -}}
</output>
{%- else -%}
<warning>
The output of your last command was too long.
Please try a different command that produces less output.
If you're looking at a file you can try use head, tail or sed to view a smaller number of lines selectively.
If you're using grep or find and it produced too much output, you can use a more selective search pattern.
If you really need to see something from the full command's output, you can redirect output to a file and then search in that file.
</warning>
{%- set elided_chars = output.output | length - 10000 -%}
<output_head>
{{ output.output[:5000] }}
</output_head>
<elided_chars>
{{ elided_chars }} characters elided
</elided_chars>
<output_tail>
{{ output.output[-5000:] }}
</output_tail>
{%- endif -%}
{% if step_limit > 0 and step_limit - n_model_calls < 20 -%}

<IMPORTANT>
There is a limit to the steps you can take. You are now {{ step_limit - n_model_calls }} steps away from reaching your limit. After you reach your limit your current solution will be auto-submitted.
At this point, please abort any specific issues that you are debugging or solving and focus on the big picture. Please make sure that
1. Your solution compiles and produces an executable (it's ok if it is still missing functionality)
2. If there are any steps left to do, or limitations that you are aware of, please write them to a document "AGENT_REPORT.md". Focus on handing off to the next agent, i.e., focus on clearly describing the problems and any todo items that are left over.
</IMPORTANT>
{%- endif %}
{% if wall_time_limit_seconds > 0 and wall_time_limit_seconds - elapsed_seconds < 600 -%}

<IMPORTANT>
You are running low on time. You have approximately {{ ((wall_time_limit_seconds - elapsed_seconds) / 60) | int }} minutes remaining before timeout.
Please wrap up your work now:
1. Ensure your solution compiles and produces an executable (it's ok if it is still missing functionality)
2. If there are any steps left to do, or limitations that you are aware of, please write them to a document "AGENT_REPORT.md". Focus on handing off to the next agent, i.e., focus on clearly describing the problems and any todo items that are left over.
3. Submit with `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
</IMPORTANT>
{%- endif %}
"""

FORMAT_ERROR_TEMPLATE = """{% if finish_reason is defined and (finish_reason == "length" or (finish_reason == "tool_calls" and not has_tool_calls)) -%}
Your previous response reached the output token limit (finish_reason={{ finish_reason }}) before you produced a tool call, so it was cut off. Respond more concisely and finish with exactly one bash tool call. If you need to think more, do so briefly.
{%- else -%}
Tool call error:

<error>
{{error}}
</error>

Here is general guidance on how to submit correct toolcalls:

Every response needs to use the 'bash' tool at least once to execute commands.

Call the bash tool with your command as the argument:
- Tool: bash
- Arguments: {"command": "your_command_here"}

If you have completed your assignment, please consult the first message about how to
submit your solution (you will not be able to continue working on this task after that).
{%- endif %}
"""


# ---- vendored mini-swe-agent -------------------------------------------------------------------

def load_minisweagent() -> SimpleNamespace:
    """The pieces of mini-swe-agent this participant uses, from the vendored copy in
    `swarmlab/vendor/minisweagent/`; RuntimeError when its version is not `MSA_VERSION`."""
    if msa_version != MSA_VERSION:
        raise RuntimeError(f"the mini_swe_agent participant follows mini-swe-agent {MSA_VERSION}, "
                           f"the vendored copy is {msa_version}")
    return SimpleNamespace(
        DefaultAgent=DefaultAgent, InterruptAgentFlow=msa_exceptions.InterruptAgentFlow,
        FormatError=msa_exceptions.FormatError, Submitted=msa_exceptions.Submitted,
        TimeExceeded=msa_exceptions.TimeExceeded,
        parse_toolcall_actions=parse_toolcall_actions,
        format_observations=format_toolcall_observation_messages)


# ---- template helpers ---------------------------------------------------------------------------

def load_templates(templates: str) -> dict[str, str]:
    """`{"system_template", "instance_template"}` of a built-in set or a `file:<path>` YAML."""
    if templates in TEMPLATE_SETS:
        path = TEMPLATES_DIR / f"{templates}.yaml"
    elif isinstance(templates, str) and templates.startswith("file:"):
        path = Path(templates[len("file:"):])
    else:
        raise ValueError(f"templates must be one of {TEMPLATE_SETS} or 'file:<path>', "
                         f"got {templates!r}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or not all(
            isinstance(data.get(k), str) for k in ("system_template", "instance_template")):
        raise ValueError(f"{path}: a template file is a mapping with string keys "
                         "system_template and instance_template")
    return {"system_template": data["system_template"],
            "instance_template": data["instance_template"]}


def resource_sentence_of(description: str) -> str:
    """The last paragraph of a ProgramBench description when it is the resource sentence."""
    last = (description or "").rsplit("\n\n", 1)[-1].strip()
    return last if last.startswith("This machine has") else ""


def rules_block(system: str) -> str:
    """The `<RULES>...</RULES>` block of the official system text."""
    start, end = system.find("<RULES>"), system.find("</RULES>")
    if start < 0 or end < 0:
        raise ValueError("the ProgramBench system text has no <RULES> block")
    return system[start:end + len("</RULES>")]


_SUBMIT_BULLET = re.compile(r"^- Submit your changes.*\n(?:  .*\n)*", re.MULTILINE)


def command_rules(instance: str) -> str:
    """The official instance text from `## Command Execution Rules` on, without the submission
    bullet (a worker never submits)."""
    start = instance.find("## Command Execution Rules")
    if start < 0:
        raise ValueError("the ProgramBench instance text has no '## Command Execution Rules'")
    text = _SUBMIT_BULLET.sub("", instance[start:])
    if SUBMIT_MARKER in text:
        raise ValueError("the submission bullet of the ProgramBench instance text changed")
    return text


def _post_channels(tools: list[ToolSchema]) -> list[str]:
    for t in tools:
        if t.name == "post":
            enum = (t.parameters.get("properties") or {}).get("channel", {}).get("enum")
            return [str(c) for c in enum or []]
    return []


def _delivery_key(item: dict) -> list:
    return [int(item.get("eligible_round") or 0), str(item.get("delivery_id") or "")]


def _strip_timestamps(messages: list[dict]) -> list[dict]:
    for m in messages:
        extra = m.get("extra")
        if isinstance(extra, dict):
            extra.pop("timestamp", None)
    return messages


# ---- the turn's transient state and the adapters --------------------------------------------------

class _Aborted(Exception):
    """The turn was cancelled: the step thread's adapter call was cancelled or refused."""


@dataclass
class _Turn:
    usage: LLMTurnUsage
    loop: asyncio.AbstractEventLoop | None
    tools: AgentTools | None
    offered: list[ToolSchema]
    turn_based: bool = False  # commit: async (`view.turn` set)
    abort: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    task: asyncio.Task | None = None  # the bridge operation running on the loop, if any
    end: bool = False
    order: list[str] = field(default_factory=list)       # call ids of the current reply
    pending: list[ToolCall] = field(default_factory=list)  # its non-bash calls not yet run
    answered: dict[str, dict] = field(default_factory=dict)  # call id -> tool message
    outputs: dict[str, dict] = field(default_factory=dict)  # bash call id -> its output

    def run(self, coro: Any) -> Any:
        """Run a coroutine on the turn's event loop from the step thread and wait for it (module
        doc, Bridge). Refused once the turn is cancelled."""
        with self.lock:  # cancel() cannot slip in between the check and the scheduling
            if self.loop is None or self.abort.is_set():
                coro.close()
                raise _Aborted("the turn was cancelled" if self.loop else "no event loop (preview)")
            fut = asyncio.run_coroutine_threadsafe(self._tracked(coro), self.loop)
        try:
            return fut.result()
        except concurrent.futures.CancelledError:
            raise _Aborted("the turn was cancelled") from None

    async def _tracked(self, coro: Any) -> Any:
        """The bridge operation on the loop: recorded in `task` so `cancel()` can cancel it."""
        with self.lock:
            if self.abort.is_set():  # cancelled between scheduling and start
                coro.close()
                raise _Aborted("the turn was cancelled")
            self.task = asyncio.current_task()
        try:
            return await coro
        finally:
            with self.lock:
                self.task = None

    def cancel(self) -> None:
        """On the loop: refuse further bridge operations and cancel the running one, at most
        once. Its `CancelledError` reaches `tools.infer`/`tools.call` (the gate releases the
        reservation); the step thread then gets `_Aborted` once that task has finished. A task
        that already has a cancellation pending (the executor's `cancel()` reached it first, or
        an earlier call here) is not cancelled again: a second `CancelledError` would interrupt
        an async world tool's cancellation cleanup, which may await (world/base.py, Async
        tools); the caller waits for the task to finish instead (`MiniSweAgent._settle`)."""
        with self.lock:
            self.abort.set()
            task = self.task
        if task is not None and not task.done() and not task.cancelling():
            task.cancel()


class _Model:
    """mini-swe-agent's Model protocol over `tools.infer` (module doc, Model adapter)."""

    def __init__(self, owner: MiniSweAgent, turn: _Turn, msa: SimpleNamespace) -> None:
        self.owner, self.turn, self.msa = owner, turn, msa
        self.config = {"observation_template": OBSERVATION_TEMPLATE,
                       "format_error_template": FORMAT_ERROR_TEMPLATE}

    # -- requests
    @staticmethod
    def _chat_message(m: dict) -> ChatMessage | None:
        role = m.get("role")
        content = m.get("content")
        text = content if isinstance(content, str) else ("" if content is None else str(content))
        if role == "assistant":
            calls = [ToolCall(call_id=tc["id"], name=tc["function"]["name"],
                              args=parse_json_args(tc["function"].get("arguments"))[0])
                     for tc in m.get("tool_calls") or []]
            return ChatMessage(role="assistant", content=text, tool_calls=calls or None,
                               provider_content=m.get("provider_content"))
        if role == "tool":
            return ChatMessage(role="tool", content=text, tool_call_id=m.get("tool_call_id"))
        if role in ("system", "user"):
            return ChatMessage(role=role, content=text)
        return None  # `exit`

    def request(self, messages: list[dict]) -> ChatRequest:
        chat = [c for c in (self._chat_message(m) for m in messages) if c is not None]
        return ChatRequest(messages=chat, tools=self.turn.offered,
                           **self.owner.model_request_defaults())

    def _format_error(self, error: str, finish: str, has_tool_calls: bool) -> Exception:
        from jinja2 import StrictUndefined, Template

        return self.msa.FormatError({
            "role": "user",
            "content": Template(FORMAT_ERROR_TEMPLATE, undefined=StrictUndefined).render(
                actions=[], error=error, has_tool_calls=has_tool_calls, finish_reason=finish),
            "extra": {"interrupt_type": "FormatError"}})

    # -- the protocol
    def query(self, messages: list[dict], **kwargs: Any) -> dict:
        resp: ChatResponse = self.turn.run(self.turn.tools.infer(self.request(messages)))
        u = self.turn.usage
        u.prompt_tokens += resp.usage.prompt_tokens
        u.completion_tokens += resp.usage.completion_tokens
        u.cost_usd += resp.cost_usd
        u.finish_reasons.append(resp.finish_reason)
        finish = FINISH_REASONS.get(resp.finish_reason, resp.finish_reason)
        summary = {"model": resp.model, "provider": resp.provider, "finish_reason": finish,
                   "usage": resp.usage.model_dump(), "cached": resp.cached}
        cost = {"cost": resp.cost_usd}
        offered = {t.name for t in self.turn.offered}
        try:
            parsed = [c for c in resp.tool_calls if c.name == "bash" or c.name not in offered]
            if "bash" not in offered and any(c.name == "bash" for c in resp.tool_calls):
                raise self._format_error("Unknown tool 'bash'.", finish, True)
            if resp.tool_calls and not parsed:
                actions = []
            else:
                actions = self.msa.parse_toolcall_actions(
                    [SimpleNamespace(id=c.call_id, function=SimpleNamespace(
                        name=c.name, arguments=_arguments(c.args))) for c in parsed],
                    format_error_template=FORMAT_ERROR_TEMPLATE,
                    template_kwargs={"finish_reason": finish})
        except self.msa.FormatError as e:
            e.messages[0]["extra"].update(cost)
            e.messages[0]["extra"]["response"] = summary
            raise
        self.turn.order = [c.call_id for c in resp.tool_calls]
        self.turn.pending = [c for c in resp.tool_calls if c.name != "bash"]
        self.turn.answered = {}
        self.turn.outputs = {}
        message: dict[str, Any] = {
            "role": "assistant", "content": resp.text or "",
            "tool_calls": [{"id": c.call_id, "type": "function",
                            "function": {"name": c.name, "arguments": _arguments(c.args)}}
                           for c in resp.tool_calls] or None}
        if resp.provider_content is not None:
            message["provider_content"] = resp.provider_content
        message["extra"] = {"actions": actions, "response": summary, **cost}
        return message

    def format_message(self, **kwargs: Any) -> dict:
        return dict(kwargs)

    def format_observation_messages(self, message: dict, outputs: list[dict],
                                    template_vars: dict | None = None) -> list[dict]:
        self.owner._run_pending(self.turn, None)
        actions = message.get("extra", {}).get("actions", [])
        observed = {m["tool_call_id"]: m for m in _strip_timestamps(self.msa.format_observations(
            actions=actions, outputs=outputs, observation_template=OBSERVATION_TEMPLATE,
            template_vars=template_vars))}
        return [observed.get(cid) or self.turn.answered[cid] for cid in self.turn.order
                if cid in observed or cid in self.turn.answered]

    def get_template_vars(self, **kwargs: Any) -> dict:
        return dict(self.config)

    def serialize(self) -> dict:
        return {"info": {"config": {"model": {"model_name": self.owner.model, **self.config},
                                    "model_type": "swarmlab tools.infer"}}}


class _Env:
    """mini-swe-agent's Environment protocol over the executor's `bash` (module doc)."""

    def __init__(self, owner: MiniSweAgent, turn: _Turn, msa: SimpleNamespace) -> None:
        self.owner, self.turn, self.msa = owner, turn, msa
        self.config: dict = {}

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
        turn = self.turn
        self.owner._run_pending(turn, action.get("tool_call_id"))
        res = turn.run(turn.tools.call("bash", {"command": action["command"]}))
        turn.usage.calls += 1
        result = res.result or {}
        if res.ok and result.get("accepted"):
            out = dict(result.get("feedback") or {})
        else:
            error = res.error if not res.ok else (result.get("feedback") or {}).get("error")
            error = error or "rejected"
            if error == "wall_limit":
                raise self.msa.TimeExceeded({"role": "exit", "content": "TimeExceeded", "extra": {
                    "exit_status": "TimeExceeded", "submission": ""}})
            if error == "submitted":
                raise self.msa.Submitted({"role": "exit", "content": "", "extra": {
                    "exit_status": "Submitted", "submission": ""}})
            if error == "turn_ended":
                turn.end = True
            out = {"output": "", "returncode": -1, "exception_info": REFUSED_PREFIX + error}
        out.setdefault("output", "")
        out.setdefault("returncode", -1)
        out.setdefault("exception_info", "")
        turn.outputs[str(action.get("tool_call_id"))] = out
        submission = submission_of(out["output"], out["returncode"])
        if submission is not None:
            raise self.msa.Submitted({"role": "exit", "content": submission, "extra": {
                "exit_status": "Submitted", "submission": submission}})
        return out

    def get_template_vars(self, **kwargs: Any) -> dict:
        return platform.uname()._asdict()

    def serialize(self) -> dict:
        return {"info": {"config": {"environment": {},
                                    "environment_type": "swarmlab executor bash"}}}


def _arguments(args: dict) -> str:
    if set(args) == {"_raw"}:
        return str(args["_raw"])
    return json.dumps(args)


# ---- the participant --------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    "steps_per_turn": None, "step_limit": 1000, "wall_limit_s": 21600, "max_tokens": 8192,
    "temperature": None, "extra": None, "templates": "solo", "idle_until_assigned": False,
    "wait_tool": False, "lead_channel": "lead", "worker_channels": None,
}


def _positive_int(name: str, value: Any, allow_none: bool = False) -> None:
    if value is None and allow_none:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1{' or None' if allow_none else ''}, "
                         f"got {value!r}")


class MiniSweAgent(Participant):
    entry_point: ClassVar[str | None] = "mini_swe_agent"
    description: ClassVar[str] = ("mini-swe-agent 2.4.6, the ProgramBench baseline agent, driven "
                                  "step by step through swarmlab's tools and providers")

    def __init__(
        self,
        model: str,
        steps_per_turn: int | None = None,
        step_limit: int = 1000,
        wall_limit_s: float = 21600,
        max_tokens: int = 8192,
        temperature: float | None = None,
        extra: dict | None = None,
        templates: str = "solo",
        idle_until_assigned: bool = False,
        wait_tool: bool = False,
        lead_channel: str = "lead",
        worker_channels: list[str] | None = None,
    ) -> None:
        from ..providers.base import split_model

        split_model(model)
        _positive_int("steps_per_turn", steps_per_turn, allow_none=True)
        _positive_int("step_limit", step_limit)
        _positive_int("max_tokens", max_tokens)
        if isinstance(wall_limit_s, bool) or not isinstance(wall_limit_s, (int, float)) \
                or wall_limit_s <= 0:
            raise ValueError(f"wall_limit_s must be a positive number, got {wall_limit_s!r}")
        if templates not in TEMPLATE_SETS and not (isinstance(templates, str)
                                                   and templates.startswith("file:")):
            raise ValueError(f"templates must be one of {TEMPLATE_SETS} or 'file:<path>', "
                             f"got {templates!r}")
        for name, flag in (("idle_until_assigned", idle_until_assigned), ("wait_tool", wait_tool)):
            if type(flag) is not bool:
                raise ValueError(f"{name} must be true or false")
        if wait_tool and templates == "solo":
            raise ValueError("wait_tool needs the lead (or a file:) templates: the solo template "
                             "offers only bash")
        if not isinstance(lead_channel, str) or not lead_channel:
            raise ValueError(f"lead_channel must be a channel name, got {lead_channel!r}")
        if worker_channels is not None and (isinstance(worker_channels, str) or not all(
                isinstance(c, str) and c for c in worker_channels)):
            raise ValueError(f"worker_channels must be a list of channel names, "
                             f"got {worker_channels!r}")
        params = getattr(self, "params", {})
        for k, default in _DEFAULTS.items():  # in params (and the spec hash) only when set
            if k in params and params[k] == default and type(params[k]) is type(default):
                params.pop(k)
        if not params.get("extra", True):  # {} is the default too
            params.pop("extra")
        self.model = model
        self.steps_per_turn = steps_per_turn
        self.step_limit = step_limit
        self.wall_limit_s = wall_limit_s
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra = dict(extra or {})
        self.templates = templates
        self.idle_until_assigned = idle_until_assigned
        self.wait_tool = wait_tool
        self.lead_channel = lead_channel
        self.worker_channels = list(worker_channels) if worker_channels is not None else None
        # ---- episode state (plain data, snapshotted by Persistable) ----
        self.messages: list[dict] = []
        self.n_calls = 0
        self.cost = 0.0
        self.n_consecutive_format_errors = 0
        self.elapsed_s = 0.0
        self.finished = False
        self.assigned = False
        self.delivered_upto: list | None = None

    # ---- swarmlab hooks ----------------------------------------------------------------------
    @property
    def calls_per_turn_cap(self) -> int:
        """Most model calls a turn can make (for `swarmlab estimate`)."""
        return self.steps_per_turn or self.step_limit

    def model_request_defaults(self) -> dict:
        return {"model": self.model, "temperature": self.temperature,
                "max_tokens": self.max_tokens, "thinking_budget": None, "extra": dict(self.extra)}

    def apply_role(self, role: Any) -> None:
        """M3c: a role's `model`, `budget.max_tokens`, `system_prompt` (replaces the system
        template) and `prompt_append` (added to the system message)."""
        self._role_prompt_append = role.prompt_append
        self._role_system_template = role.system_prompt
        self._role_name = role.name
        if role.model is not None:
            self.model = role.model
        if role.budget and "max_tokens" in role.budget:
            self.max_tokens = int(role.budget["max_tokens"])

    @property
    def exit_status(self) -> str | None:
        """The episode's exit status (`Submitted`, `LimitsExceeded`, ...), None while running."""
        if self.messages and self.messages[-1].get("role") == "exit":
            return self.messages[-1].get("extra", {}).get("exit_status")
        return None

    # ---- templates ------------------------------------------------------------------------------
    def offered_tools(self, view: View) -> list[ToolSchema]:
        """The tools the model is offered (module doc, Model adapter); `swarmlab prompts` shows
        these. The solo template: `OFFICIAL_BASH` alone when the view has `bash`, as the
        baseline. Other templates: the view's tools plus `wait_for_workers` when `wait_tool`."""
        if self.templates == "solo":
            return ([OFFICIAL_BASH.model_copy(deep=True)]
                    if any(t.name == "bash" for t in view.tools) else [])
        # the core `wait` is the participant's own (idle worker, `wait_for_workers`), never the
        # model's
        tools = [t for t in view.tools if t.name != "wait"]
        if self.wait_tool and all(t.name != WAIT_TOOL for t in tools):
            tools.append(WAIT_SCHEMA_ASYNC if view.turn is not None else WAIT_SCHEMA)
        return tools

    def template_vars(self, view: View, offered: list[ToolSchema]) -> dict:
        """The variables the templates get on top of mini-swe-agent's own (module doc)."""
        system, instance = prompt_parts(platform.uname()._asdict())
        channels = (list(self.worker_channels) if self.worker_channels is not None
                    else [c for c in _post_channels(offered) if c != self.lead_channel])
        return {"task": "", "programbench_system": system, "programbench_instance": instance,
                "programbench_rules": rules_block(system),
                "programbench_command_rules": command_rules(instance),
                "resource_sentence": resource_sentence_of(view.description),
                "lead_channel": self.lead_channel, "worker_channels": channels,
                "wait_tool": self.wait_tool, "turn_based": view.turn is not None,
                "agent": str(getattr(self, "agent", "")),
                "role": getattr(self, "_role_name", None) or "",
                "tools": [t.name for t in offered]}

    def _config(self) -> dict:
        tpl = load_templates(self.templates)
        system = getattr(self, "_role_system_template", None)
        if system is not None:
            tpl["system_template"] = system
        return {**tpl, "step_limit": self.step_limit, "cost_limit": 0,
                "wall_time_limit_seconds": int(self.wall_limit_s)}

    def _build(self, view: View, turn: _Turn, msa: SimpleNamespace) -> Any:
        """A `DefaultAgent` over this turn's adapters, from the stored state; on the first turn
        the system and instance messages as `run()` adds them."""
        agent = msa.DefaultAgent(_Model(self, turn, msa), _Env(self, turn, msa), **self._config())
        agent.extra_template_vars |= self.template_vars(view, turn.offered)
        if not self.messages:
            agent.add_messages(*self._first_messages(agent))
            if self.elapsed_s:  # async: an idle worker's waits before its first assignment
                agent._start_time = time.time() - self.elapsed_s
        else:
            agent.messages = self.messages
            agent.n_calls = self.n_calls
            agent.cost = self.cost
            agent.n_consecutive_format_errors = self.n_consecutive_format_errors
            agent._start_time = time.time() - self.elapsed_s
        return agent

    def _first_messages(self, agent: Any) -> list[dict]:
        system = agent._render_template(agent.config.system_template)
        append = getattr(self, "_role_prompt_append", None)
        if append:
            system += "\n\n" + append.strip()
        return [agent.model.format_message(role="system", content=system),
                agent.model.format_message(role="user", content=agent._render_template(
                    agent.config.instance_template))]

    def _preview(self, view: View) -> list[dict]:
        msa = load_minisweagent()
        turn = _Turn(usage=LLMTurnUsage(), loop=None, tools=None, offered=self.offered_tools(view))
        agent = msa.DefaultAgent(_Model(self, turn, msa), _Env(self, turn, msa), **self._config())
        agent.extra_template_vars |= self.template_vars(view, turn.offered)
        return self._first_messages(agent)

    def _render_system(self, view: View) -> str:
        """The first (system) message (`swarmlab prompts`, preflight)."""
        return self._preview(view)[0]["content"]

    def turn_message(self, view: View) -> ChatMessage:
        """The instance message the episode starts with (`swarmlab prompts`)."""
        return ChatMessage(role="user", content=self._preview(view)[1]["content"])

    # ---- deliveries ----------------------------------------------------------------------------
    def _new_deliveries(self, items: list[dict]) -> list[dict]:
        upto = self.delivered_upto
        return sorted((i for i in items if isinstance(i, dict)
                       and (upto is None or _delivery_key(i) > list(upto))), key=_delivery_key)

    # ---- the turn ------------------------------------------------------------------------------
    def _check_wait(self, view: View) -> None:
        """Under `commit: async` the idle worker and `wait_for_workers` need the core `wait`."""
        flags = [f for f in ("idle_until_assigned", "wait_tool") if getattr(self, f)]
        if view.turn is not None and flags and all(t.name != "wait" for t in view.tools):
            raise ValueError(WAIT_NOT_LISTED.format(flag=" and ".join(flags), agent=view.agent))

    async def _await_assignment(self, view: View, tools: AgentTools,
                                usage: LLMTurnUsage) -> list[dict]:
        """An idle worker's own board check, with no model call: under `commit: async` one
        `wait` (parked until a post arrives, or `timed_out` at once when nobody can post), its
        time on the agent's clock; otherwise one `read_board` when the agent has it."""
        if view.turn is not None:
            t0 = time.time()
            res = await tools.call("wait", {})
            usage.calls += 1
            self.elapsed_s += time.time() - t0
            items = (res.result or {}).get("posts") if res.ok else None
        elif any(t.name == "read_board" for t in view.tools):
            res = await tools.call("read_board", {})
            usage.calls += 1
            items = (res.result or {}).get("items") if res.ok else None
        else:
            return []
        return self._new_deliveries(list(items or []))

    async def turn(self, view: View, tools: AgentTools) -> LLMTurnUsage:
        usage = LLMTurnUsage()
        if self.finished:
            usage.notes.append("finished")
            return usage
        self._check_wait(view)
        msa = load_minisweagent()
        new = self._new_deliveries(list(view.pushed))
        if self.idle_until_assigned and not self.assigned:
            if not new:
                new = await self._await_assignment(view, tools, usage)
            if not new:
                usage.notes.append("idle")
                return usage
            self.assigned = True
        turn = _Turn(usage=usage, loop=asyncio.get_running_loop(), tools=tools,
                     offered=self.offered_tools(view), turn_based=view.turn is not None)
        agent = self._build(view, turn, msa)
        if new:
            agent.add_messages({"role": "user", "content": render_delivered(
                new, turn_based=view.turn is not None)})
            self.delivered_upto = _delivery_key(new[-1])
        worker = asyncio.ensure_future(asyncio.to_thread(self._steps, agent, turn, msa))
        try:
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                await self._settle(turn, worker)
                raise
        except BaseException:
            if agent.messages[-1].get("role") != "exit":
                self._repair(agent, turn, msa)
            raise
        finally:
            self._save(agent)
            status = self.exit_status
            if status is not None:
                usage.notes.append(f"exit:{status}")
        return usage

    @staticmethod
    async def _settle(turn: _Turn, worker: asyncio.Future) -> None:
        """The turn was cancelled: cancel the bridge operation (once: `_Turn.cancel`) and wait
        until the step thread has returned (so nothing touches the messages afterwards). The
        step thread returns only once the bridge task has finished, the cancelled world tool's
        cleanup included, so the participant's state is saved after that cleanup. A repeated
        cancellation while waiting keeps waiting (it does not cancel the bridge task again); the
        caller re-raises the cancellation."""
        turn.cancel()
        while not worker.done():
            try:
                await asyncio.wait({worker})
            except asyncio.CancelledError:
                turn.cancel()
        if not worker.cancelled():
            worker.exception()  # retrieved: `_Aborted` (or the step's own error) is expected

    def _steps(self, agent: Any, turn: _Turn, msa: SimpleNamespace) -> None:
        """Up to `steps_per_turn` steps, handling `InterruptAgentFlow` as `DefaultAgent.run()`."""
        done = 0
        while self.steps_per_turn is None or done < self.steps_per_turn:
            done += 1
            try:
                agent.step()
                agent.n_consecutive_format_errors = 0
            except msa.FormatError as e:
                agent.cost += e.messages[0].get("extra", {}).get("cost", 0.0)
                agent.n_consecutive_format_errors += 1
                limit = agent.config.max_consecutive_format_errors
                if 0 < limit <= agent.n_consecutive_format_errors:
                    agent.add_messages(*e.messages, {
                        "role": "exit", "content": "RepeatedFormatError",
                        "extra": {"exit_status": "RepeatedFormatError", "submission": ""}})
                else:
                    agent.add_messages(*e.messages)
            except msa.InterruptAgentFlow as e:
                agent.add_messages(*e.messages)
            if agent.messages[-1].get("role") == "exit" or turn.end:
                return

    def _run_pending(self, turn: _Turn, before: str | None) -> None:
        """Run the reply's non-bash calls issued before call `before` (all when None)."""
        limit = turn.order.index(before) if before in turn.order else len(turn.order)
        while turn.pending and turn.order.index(turn.pending[0].call_id) < limit:
            call = turn.pending.pop(0)
            turn.answered[call.call_id] = self._run_other(turn, call)

    def _run_other(self, turn: _Turn, call: ToolCall) -> dict:
        """A non-bash call of the reply, as a `tool` message (module doc, Model adapter)."""
        def message(content: str) -> dict:
            return {"role": "tool", "tool_call_id": call.call_id, "content": content,
                    "extra": {"tool": call.name}}

        if set(call.args) == {"_raw"}:
            return message(json.dumps({"ok": False, "result": {}, "error": (
                "arguments are not a JSON object; nothing was executed")}, sort_keys=True))
        if call.name == WAIT_TOOL and not turn.turn_based:
            turn.end = True
            return message(WAIT_RESULT)
        if call.name == WAIT_TOOL:  # async: the core `wait`; the reports are its result
            res = turn.run(turn.tools.call("wait", {}))
            turn.usage.calls += 1
            result = res.result or {}
            if res.ok and result.get("timed_out") and not result.get("posts"):
                return message(WAIT_STALLED)
            if res.ok:  # rendered as a `read_board` result
                return message(json.dumps({"ok": True, "result": {"items": result.get("posts")},
                                           "error": None}, sort_keys=True))
            return message(json.dumps({"ok": False, "result": res.result, "error": res.error},
                                      sort_keys=True))
        res = turn.run(turn.tools.call(call.name, call.args))
        turn.usage.calls += 1
        if call.name == "end_turn" and res.ok:
            turn.end = True
        if not res.ok and res.error == "turn_ended":
            turn.end = True
        if (call.name == "post" and res.ok and self.idle_until_assigned
                and call.args.get("channel") == self.lead_channel):
            self.assigned = False  # the report ends the assignment
            turn.end = True
        return message(json.dumps({"ok": res.ok, "result": res.result, "error": res.error},
                                  sort_keys=True))

    def _repair(self, agent: Any, turn: _Turn, msa: SimpleNamespace) -> None:
        """After an exception mid-step: answer every call of the last reply that has no result,
        with its own result when the call ran, else the baseline's not-executed observation."""
        last = next((i for i in range(len(agent.messages) - 1, -1, -1)
                     if agent.messages[i].get("role") == "assistant"), None)
        if last is None:
            return
        have = {m.get("tool_call_id") for m in agent.messages[last + 1:]}
        for tc in agent.messages[last].get("tool_calls") or []:
            cid = tc["id"]
            if cid in have:
                continue
            msg = turn.answered.get(cid) or _strip_timestamps(msa.format_observations(
                actions=[{"command": "", "tool_call_id": cid}],
                outputs=[turn.outputs.get(cid, NOT_EXECUTED)],
                observation_template=OBSERVATION_TEMPLATE,
                template_vars=agent.get_template_vars()))[0]
            agent.messages.append(msg)

    def _save(self, agent: Any) -> None:
        self.messages = agent.messages
        self.n_calls = agent.n_calls
        self.cost = agent.cost
        self.n_consecutive_format_errors = agent.n_consecutive_format_errors
        self.elapsed_s = time.time() - agent._start_time
        self.finished = self.exit_status is not None
