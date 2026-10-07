"""`swarmlab preflight SPEC [--arm A] [--seed N] [--max-usd 0.05]`: one real request per LLM
participant group, through the real provider, before any run spends (field notes item 2).

A spec whose every turn errors (e.g. a provider rejecting a body field in `extra`) still runs to
`max_rounds` at $0; preflight finds that with one cheap call per group instead.

For each participant group (as `swarmlab prompts` groups them) whose participant calls a model,
the request is exactly what the group's first agent would send in round 1, except the user
message: the agent's rendered system prompt (`_render_system` of the round-1 view), one user
line (`PROMPT`), the tools the runner would offer that agent (the world's tool schemas plus the
board tools, `RoundExecutor.schemas`), the agent's `tool_protocol`, and its
`model_request_defaults()` (model, temperature, max_tokens, thinking_budget and the exact
`extra`). The request goes straight to the provider the run would use
(`Experiment.provider_for`; no ledger, gate or cache, nothing written to disk).

Before anything is sent, the worst-case cost of all the requests (`Provider.max_cost`) is
computed; it is printed first, and when it exceeds `max_usd` nothing is sent (`PreflightRefused`).

Per group the result has: model, served_by, latency_s, finish_reason, `tool_call` (whether a tool
call parsed: native tool calls without `bad_tool_args`, or the JSON protocol's block),
`tool_calls` (names), reasoning_tokens (None when the provider reports none), cost_usd, and on
failure `error` (the provider's message, which carries the HTTP status and response text).
A group fails when the request raises or no tool call parsed; `ok` is False when any group
fails. Groups without a model (scripted participants) are listed with `skipped`.
"""
from __future__ import annotations

import asyncio
from typing import Any

from .experiment import Experiment, participant_model
from .participants.llm import ToolJsonError, parse_tool_json
from .prompts_cmd import group_views
from .providers.base import ChatMessage, ChatRequest
from .runner import _run_coro

PROMPT = "Preflight check: call one of your tools now (for example end_turn), once."


class PreflightRefused(RuntimeError):
    """The worst-case cost of the preflight requests is above `max_usd`."""


def plan(exp: Experiment, seed: int = 0) -> list[dict[str, Any]]:
    """One entry per participant group: `group, agent, count, model` and, for model-backed
    groups, the `request`, its `provider` and `worst_case_usd`."""
    out = []
    for first, count, agent, p, view in group_views(exp, seed):
        row: dict[str, Any] = {"group": len(out) + 1, "agent": str(agent), "count": count,
                               "first_index": first, "model": participant_model(p)}
        render = getattr(p, "_render_system", None)
        defaults = getattr(p, "model_request_defaults", None)
        if row["model"] is None or not callable(render) or not callable(defaults):
            row["skipped"] = "no model (scripted participant)"
            out.append(row)
            continue
        req = ChatRequest(messages=[ChatMessage(role="system", content=render(view)),
                                    ChatMessage(role="user", content=PROMPT)],
                          tools=view.tools, tool_protocol=getattr(p, "tool_protocol", "native"),
                          **defaults())
        provider, _ = exp.provider_for(req.model)
        row.update(request=req, provider=provider, worst_case_usd=provider.max_cost(req),
                   extra=dict(req.extra), tools=[t.name for t in req.tools])
        out.append(row)
    return out


def worst_case(rows: list[dict[str, Any]]) -> float:
    return sum(r.get("worst_case_usd") or 0.0 for r in rows)


async def _send(row: dict[str, Any]) -> dict[str, Any]:
    req: ChatRequest = row["request"]
    try:
        resp = await row["provider"].complete(req)
    except Exception as e:  # noqa: BLE001 - report the provider's own message
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    if req.tool_protocol == "native":
        names = [tc.name for tc in resp.tool_calls]
        parsed = bool(names) and resp.finish_reason != "bad_tool_args"
    else:
        try:
            calls = parse_tool_json(resp.text) or []
        except ToolJsonError:
            calls = []
        names = [name for name, _ in calls]
        parsed = bool(names)
    out = {"ok": parsed, "served_by": resp.served_by, "latency_s": round(resp.latency_s, 3),
           "finish_reason": resp.finish_reason, "tool_call": parsed, "tool_calls": names,
           "reasoning_tokens": resp.usage.reasoning_tokens or None,
           "prompt_tokens": resp.usage.prompt_tokens,
           "completion_tokens": resp.usage.completion_tokens, "cost_usd": resp.cost_usd,
           "text": (resp.text or "")[:200]}
    if not parsed:
        out["error"] = (f"no tool call parsed (finish_reason {resp.finish_reason!r}); agents act "
                        "only through tools")
    return out


def run_preflight(rows: list[dict[str, Any]], max_usd: float) -> list[dict[str, Any]]:
    """Send every planned request (concurrently); returns JSON-able result rows."""
    total = worst_case(rows)
    if total > max_usd:
        raise PreflightRefused(f"worst case ${total:.4f} > --max-usd ${max_usd:g}; nothing sent")
    todo = [r for r in rows if "request" in r]

    async def go() -> list[dict[str, Any]]:
        return list(await asyncio.gather(*(_send(r) for r in todo)))

    results = iter(_run_coro(go()) if todo else [])
    out = []
    for r in rows:
        base = {k: r[k] for k in ("group", "agent", "count", "model") if k in r}
        if "request" not in r:
            out.append({**base, "ok": True, "skipped": r.get("skipped")})
            continue
        out.append({**base, "extra": r["extra"], "tools": r["tools"],
                    "worst_case_usd": r["worst_case_usd"], **next(results)})
    return out


def result_lines(results: list[dict[str, Any]]) -> list[str]:
    lines = []
    for r in results:
        head = f"group {r['group']} ({r['agent']} x{r['count']})"
        if r.get("skipped"):
            lines.append(f"{head}: skipped ({r['skipped']})")
            continue
        head += f" {r['model']}"
        if "latency_s" not in r:
            lines.append(f"FAIL {head}: {r['error']}  (extra={r['extra']})")
            continue
        reasoning = (f"{r['reasoning_tokens']} reasoning tokens" if r["reasoning_tokens"]
                     else "reasoning tokens not reported")
        tool = (f"tool call parsed: {', '.join(r['tool_calls'])}" if r["tool_call"]
                else "NO tool call parsed")
        lines.append(f"{'ok  ' if r['ok'] else 'FAIL'} {head}: {r['latency_s']:.2f} s, finish="
                     f"{r['finish_reason']}, {tool}, {reasoning}, "
                     f"{r['completion_tokens']} completion tokens, ${r['cost_usd']:.5f}"
                     + (f", served by {r['served_by']}" if r.get("served_by") else ""))
        if not r["ok"]:
            lines.append(f"     {r['error']}; reply text: {r['text']!r}")
    return lines
