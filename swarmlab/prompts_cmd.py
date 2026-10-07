"""`swarmlab prompts SPEC [--arm A] [--seed N]`: the exact prompts an arm's agents see, with no
model call (docs/INTERFACE-M4.md §4). Meant for the second-agent review before spending.

For each participant group (consecutive agents with the same participant spec, as `count` groups
them in YAML) the first agent of the group is shown: the world is built from the arm and reset
with the run's seed exactly as the runner does (`derive(seed, "world")`, agents `a000...`), the
participant is bound (`derive(seed, "agent", a)`), the agent's round-1 observation is taken, and
the participant's own rendering is used: `LLMAgent._render_system(view)` (the Jinja template,
plus the JSON-protocol section under `tool_protocol: json`) and `LLMAgent.round_message(view)`.

Decisions:

- The view is the round-1 view: no outcomes, and nothing pushed (pushed items exist only from
  round 2). Evaluator-only `private` data is stripped as in a run; it is not shown.
- The tools are the executor's schemas for that agent (`RoundExecutor.schemas`), i.e. what the
  provider receives under the native protocol.
- A participant without these rendering methods (scripted participants) is listed with
  `system: null` and a note; nothing else of it is rendered.
- Image parts of the user message render as `[image: <n> base64 chars]` in the text output (the
  JSON output keeps the parts as they are).
- M3c roles: groups split where the assigned role changes; every agent is bound and its role
  applied (`roles.bind_roles`, which also lays out a `Tree` topology's channels) before
  rendering, so the system prompt carries the role's `prompt_append` and the tools are the
  role's. Each row has `role` (the role name or None).
- The seed defaults to the first of the YAML's `seeds:` (0 when none); FlagGame crops depend on
  it, prompts otherwise do not.
"""
from __future__ import annotations

import copy
from typing import Any

from .executor import RoundExecutor
from .experiment import Experiment, participant_model
from .ids import agent_id
from .rng import derive
from .roles import agent_roles, bind_roles, role_name, spec_roles
from .view import View


def participant_groups(exp: Experiment) -> list[tuple[int, int]]:
    """(first agent index, count) per group of consecutive identical participant specs."""
    groups: list[tuple[int, int]] = []
    prev: Any = None
    for i, p in enumerate(exp.participants):
        spec = (p.spec(), role_name(p))
        if groups and spec == prev:
            groups[-1] = (groups[-1][0], groups[-1][1] + 1)
        else:
            groups.append((i, 1))
        prev = spec
    return groups


def _text(message: Any) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    out = []
    for p in content:
        if p.type == "image":
            out.append(f"[image: {len(p.image_png_b64 or '')} base64 chars]")
        else:
            out.append(p.text or "")
    return "\n".join(out)


def group_views(exp: Experiment, seed: int = 0) -> list[tuple[int, int, Any, Any, View]]:
    """`(first agent index, count, agent id, bound participant, round-1 view)` per participant
    group, built exactly as the runner builds them (also used by `swarmlab preflight`)."""
    agents = [agent_id(i) for i in range(len(exp.participants))]
    world = copy.deepcopy(exp.world)
    world.reset(derive(seed, "world"), list(agents))
    board = copy.deepcopy(exp.medium)
    commit = exp.options.get("commit", "round_end")
    board.commit_mode = commit
    bound = {a: copy.deepcopy(p) for a, p in zip(agents, exp.participants, strict=True)}
    for a, p in bound.items():
        p.bind(a, derive(seed, "agent", a))
    names = [role_name(p) for p in exp.participants]
    roles = bind_roles(board, bound, agent_roles(*spec_roles(exp.roles, names), len(agents)))
    ex = RoundExecutor(run="prompts", round=1, world=world, board=board, blobs=None,
                       agents=list(agents), commit=commit,
                       max_calls_per_turn=int(exp.options.get("max_calls_per_turn", 20)),
                       roles=roles)
    out = []
    for first, count in participant_groups(exp):
        a = agents[first]
        obs = world.observe(a).model_copy(update={"private": {}})
        view = View(round=1, agent=a, observation=obs, outcomes=[], pushed=[],
                    tools=ex.schemas(a), description=world.description())
        out.append((first, count, a, bound[a], view))
    return out


def render_prompts(exp: Experiment, seed: int = 0) -> list[dict]:
    """One dict per participant group: agent, type, count, model, system, user, tools."""
    agents = [agent_id(i) for i in range(len(exp.participants))]
    names = [role_name(p) for p in exp.participants]
    out = []
    for first, count, a, p, view in group_views(exp, seed):
        spec = p.spec()
        row: dict[str, Any] = {
            "group": len(out) + 1, "agent": str(a), "agents": [str(x) for x in
                                                              agents[first:first + count]],
            "type": spec["type"], "count": count, "model": participant_model(p),
            "role": names[first],
            "tool_protocol": getattr(p, "tool_protocol", None),
            "tools": [t.name for t in view.tools], "system": None, "user": None,
            "user_parts": None, "note": None,
        }
        render = getattr(p, "_render_system", None)
        round_message = getattr(p, "round_message", None)
        if callable(render) and callable(round_message):
            row["system"] = render(view)
            msg = round_message(view)
            row["user"] = _text(msg)
            row["user_parts"] = (msg.content if isinstance(msg.content, str)
                                 else [part.model_dump(mode="json") for part in msg.content])
        else:
            row["note"] = (f"{spec['type']} is not prompted: it acts by code and sees the "
                           "observation and tools directly")
        out.append(row)
    return out


def prompts_text(rows: list[dict], arm: str | None, seed: int) -> str:
    lines = [f"arm {arm}  seed {seed}  ({len(rows)} participant group(s); one agent each)"]
    for r in rows:
        span = r["agents"][0] if r["count"] == 1 else f"{r['agents'][0]}..{r['agents'][-1]}"
        model = f"  model {r['model']}" if r["model"] else ""
        model += f"  role {r['role']}" if r.get("role") else ""
        proto = f"  tool_protocol {r['tool_protocol']}" if r["tool_protocol"] else ""
        lines += ["", (f"=== group {r['group']}: {r['type']} x{r['count']} ({span}){model}{proto}"
                       f"  -- showing {r['agent']} ==="),
                  f"tools: {', '.join(r['tools'])}"]
        if r["system"] is None:
            lines.append(f"({r['note']})")
            continue
        lines += ["--- system prompt ---", r["system"], "--- round 1 user message ---", r["user"]]
    return "\n".join(lines)
