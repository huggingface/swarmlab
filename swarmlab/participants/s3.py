"""The s3 agent (arXiv 2611.08615, §3.1 and Appendices F-G): an asynchronous, token-level harness.

One participant per agent, run under `commit: async` with one turn per agent
(`rounds_per_agent: 1`): the whole episode is that turn. The agent works against a vLLM server
token by token, so the exact sampled tokens and their logprobs are kept for training (the
paper's loss covers only generated tokens and needs their sampling logprobs):

- The first prompt is the paper's system prompt (`system_prompt`) and the world's first user
  turn, rendered with the model's chat template and the paper's tools by the server's
  `/tokenize`. From then on the context only grows: every later segment (tool results,
  notifications, the final-answer request) is appended as tokens, never re-rendered.
- Generation runs in requests of at most `chunk` (1,024) new tokens through `/v1/completions`
  on the token ids, continuing from the exact prefix. At each request boundary a new message
  arrival is delivered as a notification, appended as a tool output (Appendix F: "the sampled
  prefix is kept unchanged, a notification is appended as a tool output, and the agent
  continues"); for the chat template that closes the assistant message in progress.
- A complete response with tool calls runs them and appends their results (plus a notification
  if messages arrived). A response without a tool call either answers (`#### [..]`, recorded
  with the world's `submit`) or makes the agent idle: it parks in the core `wait` (peek mode,
  nothing read) until a message it has not been told about arrives, then gets the notification.
  If no other agent can post (all idle or finished), the work phase ends.
- After the work phase (idle with nobody left to wake it, `work_tokens` spent, or the context
  full), an agent that has not answered is asked once with the paper's final-answer request and
  gets `final_tokens` more.

Tools shown to the model are the paper's (`message`, `read_messages`, `check_notifications`,
Table 2). `message(to, text)` posts on the board under the `direct` topology with
`allow_all: true` (`to` = a teammate's public name, mapped to its agent id, or `all`);
`read_messages()` returns up to 16 unread messages with their senders' public names.

State kept (snapshotted, read back for training): `tokens`, `mask` (1 = sampled by the model),
`logprobs` (sampling logprob per token, 0.0 where mask is 0), `prompt_len`, `work_generated`,
`final_generated`, `answer`, `events` (a short log of what the agent did, for analysis).
Model usage is the agent's own (`/v1/completions` is called directly, not through swarmlab's
providers, which speak chat only), so swarmlab's ledger does not count it; `turn` returns it.
"""
from __future__ import annotations

import json
import re
from typing import Any, ClassVar

import httpx

from ..tools import AgentTools
from ..view import View
from ..world.aggregation import ROSTER_PREFIX, count_words
from .base import Participant
from .llm import LLMTurnUsage

# ---- the paper's prompt texts (Appendix G; "[...]" elisions are unpublished and left out) ------
TASK = (
    "You are in a team solving a private-information number sorting task. Each of {n} agents receives "
    "one English-written number. You initially know only your own number. Share your actual assignment, "
    "explicitly read partners' messages, and associate each received number with its sender so repeated "
    "messages do not create duplicates. Collect one number per agent and sort by numeric value, retaining "
    "equal values from different agents. Every agent can submit the full list. The team earns reward if "
    "at least one final answer is correct; {reward}"
)
REWARD_TOKEN_COST = (
    "there is no message-count reward. Successful rewards are reduced by at most 0.25, in proportion to "
    "the tokens all agents generated divided by the team's total generation allowance. Failures pay no "
    "token cost, and a correct answer always earns more than a failure."
)
REWARD_NO_COST = "there is no message-count or length reward."
GUIDANCE = (
    "Agents work toward the same team objective; no specialist roles are assigned by the harness. A task "
    "may give you private information that partners do not know. Work independently when useful. "
    "Communicate when you have information, checked evidence, a counterexample, a correction, or a "
    "specific question that may change a teammate's next step. Distinguish verified facts from guesses. "
    "Read replies explicitly; sending does not read messages. Teammate names are recipients of message, "
    "never function names. Use one recipient or all. Do not count repeated claims as independent "
    "verification. If evidence conflicts, compare the underlying reasoning. Submit your own checked "
    "answer when asked. Examples are hypothetical and contain no evidence about the current problem."
)
SHARED_REWARD = (
    "All participating agents share the same reward if at least one agent active at the end submits a "
    "correct answer. There is no individual-correctness bonus."
)
IDENTITY = "Your name is {name}. Your teammates: {teammates}."  # ours: the paper does not print part 4
EXAMPLES = (  # part 5: the paper's two example calls, in the model's tool-call syntax ("Example..." is ours)
    "Example tool calls:\n"
    '<tool_call>\n{"name": "message", "arguments": {"to": "all", "text": "My private number is '
    'forty-two (42). Please share your own assignment."}}\n</tool_call>\n'
    '<tool_call>\n{"name": "read_messages", "arguments": {}}\n</tool_call>'
)
NOTIFICATION_RULES = (
    "While you are working, we will immediately notify you whenever another agent sends you a message, "
    "interrupting your current generation if needed. Notifications show unread-message counts by sender. "
    "If you have no unread-message notification, there is no need to check your inbox. Use read_messages "
    "when notified; do not repeatedly poll an empty inbox. You can continue reasoning or doing useful work "
    "while waiting for a notification."
)
IDLENESS_RULES = (
    "A complete tool call runs immediately and you receive its result. If you end a response without a "
    "tool call or final answer, you become idle: no tokens are spent while waiting, and a new peer message "
    "wakes you. If everyone is idle or finished, you will be asked for your final answer once. Do not "
    "repeat statements merely to keep yourself active."
)
FINAL_REQUEST = (
    "Submit the sorted list of all {n} agents' numbers now. Reply with one line: #### followed by a JSON "
    "array of {n} integers in ascending order, retaining duplicates. The messaging phase has ended."
)
NOTICE = "You have {count} unread message{s} from {senders}.\nUse read_messages to read them; " \
         "this notification does not mark messages as read."

TOOLS = [  # Table 2
    {"type": "function", "function": {
        "name": "message",
        "description": ("Send plans, questions, tentative ideas, dead ends, or checked results to one "
                        "available teammate or all available teammates. Label uncertainty and verification "
                        "status. Delivery is immediate. Sending does not read incoming messages or wait for "
                        "replies; use read_messages when notified."),
        "parameters": {"type": "object", "properties": {
            "to": {"type": "string", "description": "A teammate's name, or all."},
            "text": {"type": "string"}}, "required": ["to", "text"]}}},
    {"type": "function", "function": {
        "name": "read_messages", "description": "Read explicitly received unread peer messages.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "check_notifications",
        "description": "Check unread message counts by sender without reading messages.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]
READ_LIMIT = 16

TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
ANSWER = re.compile(r"####\s*(\[[^\]]*\])")


def system_prompt(name: str, teammates: list[str], n: int, token_cost: bool = True) -> str:
    words = count_words(n)
    return "\n\n".join([
        TASK.format(n=words, reward=REWARD_TOKEN_COST if token_cost else REWARD_NO_COST),
        GUIDANCE, SHARED_REWARD, IDENTITY.format(name=name, teammates=", ".join(teammates)),
        EXAMPLES, NOTIFICATION_RULES, IDLENESS_RULES])


def parse_answer(text: str) -> list[int] | None:
    """The last `#### [..]` JSON array of integers in `text`, or None."""
    for m in reversed(ANSWER.findall(text)):
        try:
            v = json.loads(m)
        except json.JSONDecodeError:
            continue
        if isinstance(v, list) and all(isinstance(x, int) and not isinstance(x, bool) for x in v):
            return v
    return None


def after_thinking(text: str) -> str:
    return text.rsplit("</think>", 1)[-1]


class S3Agent(Participant):
    entry_point: ClassVar[str | None] = "s3_agent"
    description: ClassVar[str] = "s3 (arXiv 2611.08615) agent: async, token-level, paper prompts and tools"
    _skip_in_snapshot: ClassVar[tuple[str, ...]] = (
        "params", "model", "base_url", "work_tokens", "final_tokens", "chunk", "temperature",
        "context", "token_cost", "timeout_s")

    def __init__(self, model: str = "Qwen/Qwen3-0.6B", base_url: str = "http://127.0.0.1:8000",
                 work_tokens: int = 8192, final_tokens: int = 1024, chunk: int = 1024,
                 temperature: float = 1.0, context: int = 40960, token_cost: bool = True,
                 enable_thinking: bool = True, timeout_s: float = 600.0) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.work_tokens, self.final_tokens, self.chunk = work_tokens, final_tokens, chunk
        self.temperature, self.context, self.token_cost = temperature, context, token_cost
        self.enable_thinking = enable_thinking
        self.timeout_s = timeout_s
        self.tokens: list[int] = []
        self.mask: list[int] = []
        self.logprobs: list[float] = []
        self.prompt_len = 0
        self.work_generated = 0
        self.final_generated = 0
        self.answer: list[int] | None = None
        self.done = False
        self.events: list[dict] = []
        self.notified: list[str] = []  # delivery ids the agent has been told about

    # ---- vLLM -----------------------------------------------------------------------------------
    async def _post(self, path: str, body: dict) -> dict:
        r = await self._http.post(f"{self.base_url}{path}", json=body)
        r.raise_for_status()
        return r.json()

    async def _encode(self, text: str) -> list[int]:
        out = await self._post("/tokenize", {"model": self.model, "prompt": text, "add_special_tokens": False})
        return list(out["tokens"])

    async def _append_env(self, text: str) -> None:
        ids = await self._encode(text)
        self.tokens += ids
        self.mask += [0] * len(ids)
        self.logprobs += [0.0] * len(ids)

    async def _generate(self, max_tokens: int) -> tuple[str, str]:
        """Sample up to `max_tokens` from the current context; append them; (text, finish_reason)."""
        out = await self._post("/v1/completions", {
            "model": self.model, "prompt": self.tokens, "max_tokens": max_tokens,
            "temperature": self.temperature, "top_p": 1.0, "logprobs": 0,
            "return_tokens_as_token_ids": True, "skip_special_tokens": False})
        choice = out["choices"][0]
        lp = choice.get("logprobs") or {}
        ids = [int(t.split(":", 1)[1]) for t in lp.get("tokens") or []]
        lps = [float(x) if x is not None else 0.0 for x in lp.get("token_logprobs") or []]
        self.tokens += ids
        self.mask += [1] * len(ids)
        self.logprobs += lps
        usage = out.get("usage") or {}
        self._usage.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self._usage.completion_tokens += len(ids)
        self._usage.calls += 1
        return choice.get("text") or "", choice.get("finish_reason") or "stop"

    # ---- board ----------------------------------------------------------------------------------
    def _name(self, agent_id: str | None) -> str:
        return self._names.get(str(agent_id), str(agent_id))

    async def _unread(self, tools: AgentTools, wait: bool) -> tuple[list[dict], bool]:
        """Unread items (`delivery_id`, `sender`), nothing marked read; with `wait`, park until one
        the agent was not told about arrives. (items, stalled)."""
        args: dict[str, Any] = {"peek": True}
        if wait:
            args["known"] = list(self.notified)
        else:
            args["timeout_s"] = 0
        res = await tools.call("wait", args)
        if not res.ok:
            raise RuntimeError(f"wait failed: {res.error}")
        return list(res.result.get("unread") or []), bool(res.result.get("timed_out")) and wait

    def _notice(self, unread: list[dict]) -> str | None:
        """The notification for `unread` when it holds an item not yet notified, else None."""
        new = [u for u in unread if u["delivery_id"] not in self.notified]
        if not new:
            return None
        self.notified += [u["delivery_id"] for u in new]
        counts: dict[str, int] = {}
        for u in unread:
            counts[self._name(u["sender"])] = counts.get(self._name(u["sender"]), 0) + 1
        total = sum(counts.values())
        senders = " and ".join(counts) if len(counts) <= 2 else ", ".join(list(counts)[:-1]) + \
            ", and " + list(counts)[-1]
        self.events.append({"kind": "notified", "at": len(self.tokens), "unread": counts})
        return json.dumps({"unread_by_sender": counts,
                           "notice": NOTICE.format(count=total, s="" if total == 1 else "s", senders=senders)})

    async def _tool(self, tools: AgentTools, call: dict) -> str:
        name, args = call.get("name"), call.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return json.dumps({"error": "arguments are not valid JSON"})
        if name == "message":
            to, text = args.get("to"), args.get("text")
            if not isinstance(to, str) or not isinstance(text, str):
                return json.dumps({"error": "message needs `to` (a teammate's name or all) and `text`"})
            target = "all" if to.strip().lower() == "all" else self._ids.get(to.strip())
            if target is None or target == str(self.agent):
                return json.dumps({"error": f"unknown recipient {to!r}; use one of {self._teammates} or all"})
            res = await tools.call("post", {"text": text, "to": target})
            self.events.append({"kind": "message", "at": len(self.tokens), "to": to, "text": text, "ok": res.ok})
            return json.dumps({"status": "sent", "to": to} if res.ok else {"error": res.error})
        if name == "read_messages":
            res = await tools.call("read_board", {"limit": READ_LIMIT})
            items = (res.result or {}).get("items") or [] if res.ok else []
            msgs = [{"from": self._name(it.get("sender")), "text": it.get("content")} for it in items]
            self.events.append({"kind": "read", "at": len(self.tokens), "n": len(msgs),
                                "from": [m["from"] for m in msgs]})
            return json.dumps({"messages": msgs})
        if name == "check_notifications":
            unread, _ = await self._unread(tools, wait=False)
            counts: dict[str, int] = {}
            for u in unread:
                counts[self._name(u["sender"])] = counts.get(self._name(u["sender"]), 0) + 1
            return json.dumps({"unread_by_sender": counts})
        return json.dumps({"error": f"unknown tool {name!r}"})

    # ---- the episode ----------------------------------------------------------------------------
    async def _setup(self, view: View) -> None:
        texts = [part.text or "" for part in view.observation.parts if part.type == "text"]
        p = json.loads(next(t for t in texts if t.startswith(ROSTER_PREFIX))[len(ROSTER_PREFIX):])
        self._ids = dict(p["ids"])
        self._names = {v: k for k, v in self._ids.items()}
        self._teammates = list(p["teammates"])
        self._n = int(p["n"])
        first = "".join(t for t in texts if not t.startswith(ROSTER_PREFIX))
        out = await self._post("/tokenize", {
            "model": self.model, "tools": TOOLS, "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
            "messages": [{"role": "system", "content": system_prompt(p["name"], self._teammates, self._n,
                                                                      self.token_cost)},
                         {"role": "user", "content": first}]})
        self.tokens = list(out["tokens"])
        self.mask = [0] * len(self.tokens)
        self.logprobs = [0.0] * len(self.tokens)
        self.prompt_len = len(self.tokens)

    async def _close_assistant(self, finish: str) -> None:
        """End the assistant message in progress (the model's own end token when it stopped)."""
        if not (finish == "stop" and self.tokens and self.mask[-1] == 1 and self._is_end(self.tokens[-1])):
            await self._append_env("<|im_end|>")
        await self._append_env("\n")

    def _is_end(self, tok: int) -> bool:
        return tok in self._end_ids

    async def turn(self, view: View, tools: AgentTools) -> LLMTurnUsage:
        self._usage = LLMTurnUsage()
        if self.done:
            self._usage.notes.append("finished")
            return self._usage
        async with httpx.AsyncClient(timeout=self.timeout_s) as http:
            self._http = http
            await self._setup(view)
            self._end_ids = set(await self._encode("<|im_end|>"))
            reason = await self._work(tools)
            self.events.append({"kind": "work_end", "at": len(self.tokens), "reason": reason})
            if self.answer is None:
                await self._final(tools)
        self.done = True
        return self._usage

    async def _work(self, tools: AgentTools) -> str:
        segment = ""  # text of the assistant message in progress
        while True:
            room = min(self.work_tokens - self.work_generated, self.context - len(self.tokens) - self.final_tokens - 256)
            if room <= 0:
                return "allowance" if self.work_tokens <= self.work_generated else "context"
            before = len(self.tokens)
            text, finish = await self._generate(min(self.chunk, room))
            self.work_generated += len(self.tokens) - before
            segment += text
            if finish == "length":  # a request boundary: deliver a notification if one is due
                unread, _ = await self._unread(tools, wait=False)
                note = self._notice(unread)
                if note is not None:
                    await self._close_assistant(finish)
                    await self._append_env(f"<|im_start|>user\n<tool_response>\n{note}\n</tool_response>"
                                           "<|im_end|>\n<|im_start|>assistant\n")
                    segment = ""
                continue
            await self._close_assistant(finish)
            calls = []
            for raw in TOOL_CALL.findall(after_thinking(segment)):
                try:
                    calls.append(json.loads(raw))
                except json.JSONDecodeError:
                    calls.append({"name": None, "arguments": {}, "_bad": raw})
            answer = parse_answer(after_thinking(segment))
            segment = ""
            if calls:
                results = [json.dumps({"error": "tool call is not valid JSON"}) if c.get("_bad") is not None
                           else await self._tool(tools, c) for c in calls]
                unread, _ = await self._unread(tools, wait=False)
                note = self._notice(unread)
                if note is not None:
                    results.append(note)
                body = "\n".join(f"<tool_response>\n{r}\n</tool_response>" for r in results)
                await self._append_env(f"<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n")
                continue
            if answer is not None:
                await self._submit(tools, answer, "work")
                return "answered"
            unread, stalled = await self._unread(tools, wait=True)  # idle until a new message
            note = self._notice(unread)
            if note is None:
                return "idle" if stalled else "woken_without_news"
            self.events.append({"kind": "woken", "at": len(self.tokens)})
            await self._append_env(f"<|im_start|>user\n<tool_response>\n{note}\n</tool_response>"
                                   "<|im_end|>\n<|im_start|>assistant\n")

    async def _final(self, tools: AgentTools) -> None:
        req = FINAL_REQUEST.format(n=count_words(self._n))
        await self._append_env(f"<|im_start|>user\n{req}<|im_end|>\n<|im_start|>assistant\n")
        before = len(self.tokens)
        text, _ = await self._generate(self.final_tokens)
        self.final_generated = len(self.tokens) - before
        await self._submit(tools, parse_answer(after_thinking(text)) or [], "final")

    async def _submit(self, tools: AgentTools, answer: list[int], phase: str) -> None:
        self.answer = answer
        res = await tools.call("submit", {"answer": answer})
        self.events.append({"kind": "submit", "at": len(self.tokens), "phase": phase, "answer": answer,
                            "ok": res.ok})
