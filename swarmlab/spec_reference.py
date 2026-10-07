"""The experiment YAML reference (`swarmlab spec-reference`) and readable shape errors.

The reference is generated from the pydantic models in `swarmlab/spec.py` (`ExperimentDoc`,
`ArmDoc`, `MediumSpec`, `ParticipantGroup`, `PluginSpec`, `Budget`, `RunOptions`) and
`swarmlab/roles.py` (`Role`): every key, its shape and its default come from the models; the
one-line meaning of each key comes from `DESCRIPTIONS` below (a test checks that every model
field has one). README.md's "Spec reference" section is this output verbatim between the
`<!-- spec-reference:start -->` / `<!-- spec-reference:end -->` markers (a test checks they agree;
regenerate with `swarmlab spec-reference --readme README.md`).

`explain_validation_error(err)` turns a pydantic `ValidationError` raised on an experiment
document into one line per problem, each naming the key path and the shape that key expects
(e.g. `arms.gossip.medium.topology: expected {type: NAME, params: {...}} ...`), followed by the
raw pydantic message.

Decisions:

- Plain `dict` fields that are validated elsewhere (`options` -> `RunOptions`, an arm's `budget`
  -> `Budget`, `roles` -> `{name: Role}`) are expanded through `EXPANDS`, so the reference shows
  their keys too.
- No shorthand such as `topology_params` is accepted; the error says what the right shape is.
"""
from __future__ import annotations

import types
import typing
from typing import Any, Literal

from pydantic import BaseModel, ValidationError
from pydantic_core import PydanticUndefined

from .roles import Role
from .spec import (
    ArmDoc,
    Budget,
    ExperimentDoc,
    MediumSpec,
    ParticipantGroup,
    PluginSpec,
    RunOptions,
    StopWhen,
)

START = "<!-- spec-reference:start -->"
END = "<!-- spec-reference:end -->"

PLUGIN_SHAPE = "{type: NAME, params: {...}}"
PLUGIN_NOTE = ('a plugin: `{type: NAME, params: {...}}`, or the bare string `NAME` for '
               '`{type: NAME, params: {}}`')

# (model, field) -> the model a plain `dict` field is validated against
EXPANDS: dict[tuple[type[BaseModel], str], tuple[str, type[BaseModel]]] = {
    (ExperimentDoc, "options"): ("mapping", RunOptions),
    (ArmDoc, "options"): ("mapping", RunOptions),
    (ArmDoc, "budget"): ("mapping", Budget),
    (ExperimentDoc, "roles"): ("by_name", Role),
    (ArmDoc, "roles"): ("by_name", Role),
}

DESCRIPTIONS: dict[type[BaseModel], dict[str, str]] = {
    ExperimentDoc: {
        "name": "experiment name; run ids are `<name>__<arm>__s<seed>`",
        "arms": "the conditions, by arm name; each arm is a full setup (see `arms.<arm>`)",
        "budget": "per-run caps in USD (each arm's `budget` is merged over it) plus `total_usd`, "
                  "the cap for the whole experiment",
        "options": "run options for every arm (each arm's `options` is merged over them)",
        "providers": "provider overrides by model prefix, e.g. `hf: {type: openai_compat, "
                     "params: {name: hf, timeout_s: 60}}`",
        "seeds": "seeds `swarmlab run` runs per arm (empty: `[0]`)",
        "interventions": "interventions for every arm (an arm's own list is appended)",
        "roles": "role declarations by name, for every arm (an arm's `roles` is merged per name)",
    },
    ArmDoc: {
        "world": "the task, e.g. `{type: flaggame, params: {n_candidates: 8}}` or `flaggame`",
        "participants": "participant groups, in agent order",
        "medium": "the message board: topology, policies, registry",
        "metrics": "metrics logged every round (`swarmlab metrics` lists them)",
        "probes": "probes asked after every commit, e.g. `[belief]`",
        "interventions": "interventions (`inject_post`, `mute`, ...) for this arm",
        "options": "run options for this arm, merged over the top-level `options`",
        "budget": "per-run caps for this arm, merged over the top-level `budget` (no `total_usd`)",
        "roles": "role declarations for this arm, merged over the top-level `roles`",
    },
    ParticipantGroup: {
        "type": "participant type: `llm`, `evidence_aggregator`, `enumerator`, `silent`, ...",
        "count": "number of agents in this group (>= 1)",
        "role": "role name (declared in `roles`, or a built-in: worker, coordinator, reviewer, "
                "skeptic, scribe, manager)",
        "params": "constructor params, e.g. `{model: \"anthropic:claude-haiku-4-5\", "
                  "max_tokens: 1024}` for `llm`",
    },
    MediumSpec: {
        "topology": "who receives a post: `broadcast`, `gossip` (params `{k: 1}`: partners per "
                    "agent per round), `groups`, `star` (params `{center: a000}`: members reach only the "
                    "center, the center reaches all), `tree`; e.g. "
                    "`{type: gossip, params: {k: 2}}`",
        "delivery": "`pull` (agents call `read_board`) or `push` (deliveries come with the turn)",
        "push_limit": "most items pushed per turn under `delivery: push`",
        "push_consume": "with `delivery: push`: show the newest `push_limit` unread items and mark "
                        "every pushed-up-to item read, so each item is pushed once",
        "policies": "visibility policies applied in order, e.g. `[{type: delay, params: "
                    "{rounds: 1}}]`",
        "channels": "board channels",
        "registry": "turn on the claim registry tools",
        "claim_policy": "`advisory` or `enforced` (registry claims checked against world actions)",
    },
    Budget: {
        "soft_usd": "end the run at the next round boundary once agent spend reaches this (0: off)",
        "hard_usd": "absolute ceiling on agent + probe spend for the run: reached mid-round, the "
                    "round is discarded (end `hard_ceiling`); reached by the probes after the "
                    "commit, the round is kept (end `hard_ceiling_probes`) (0: off)",
        "measurement_usd": "cap on probe spend (0: off)",
        "total_usd": "top level only: cap on the whole experiment's spend (all arms x seeds, "
                     "finished and resumed runs included) (0: off)",
    },
    RunOptions: {
        "seed": "set per run from `seeds` (or `--seed`); not written in the YAML",
        "max_rounds": "rounds per run (required here or as `--max-rounds`)",
        "commit": "`round_end` (phase commit) or `immediate` (sequential)",
        "max_calls_per_turn": "tool calls an agent may make per turn",
        "snapshot_every": "write a snapshot every N rounds",
        "concurrency": "concurrent turns",
        "repeat": "paired-run repeat index (0: a plain run)",
        "scheduler": "turn order per round: `seeded_shuffle` (default, every live agent) or "
                     "`one_speaker` (one random live agent per round; pair with `gossip` k=1 and "
                     "`commit: immediate` for the Flag Game paper's pairwise protocol)",
        "rounds_per_agent": "sugar: `max_rounds = rounds_per_agent x number of agents`",
        "stop_when": "`{metric, op, value, consecutive}`: end the run (`stop_condition`) when "
                     "the metric compares true at `consecutive` evaluations in a row (probe "
                     "rounds when the run has probes, else every round)",
    },
    StopWhen: {
        "metric": "a logged metric name, e.g. `belief.consensus@probe:belief`",
        "op": "comparison: `>=`, `>`, `<=`, `<` or `==`",
        "value": "threshold the metric is compared with",
        "consecutive": "evaluations in a row the comparison must hold",
    },
    Role: {
        "name": "set from the key under `roles:`",
        "prompt_append": "text appended to the system prompt",
        "system_prompt": "replaces the system prompt",
        "tools": "allowlist of tool names (null: all)",
        "channels_read": "channels the role reads (null: all)",
        "channels_write": "channels the role writes (null: all)",
        "registry": "registry access: `none`, `read`, `write`",
        "may_act": "may use the world's action tools",
        "model": "overrides the participant's model (`prefix:id`)",
        "budget": "per-turn overrides: `{max_calls, max_tokens}`",
        "post_fields": "allowed post fields and values: `{field: [values]}`",
    },
}

SKIP_FIELDS = {(RunOptions, "seed"), (Role, "name")}


def _is_plugin(tp: Any) -> bool:
    return tp is PluginSpec


def shape(tp: Any) -> str:
    """A short YAML-ish shape for a type annotation."""
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if _is_plugin(tp):
        return PLUGIN_SHAPE
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        keys = [k for k in tp.model_fields if (tp, k) not in SKIP_FIELDS]
        return "{" + ", ".join(keys) + "}"
    if origin in (list, tuple):
        return "[" + (shape(args[0]) if args else "...") + ", ...]"
    if origin is dict:
        return "{" + (f"NAME: {shape(args[1])}" if args else "...") + "}"
    if origin is Literal:
        return " | ".join(str(a) for a in args)
    if origin in (typing.Union, types.UnionType):
        return " | ".join("null" if a is type(None) else shape(a) for a in args)
    if tp is dict:
        return "{...}"
    return getattr(tp, "__name__", str(tp))


def _default(info: Any) -> str:
    if info.is_required():
        return "required"
    d = info.default
    if d is PydanticUndefined:
        d = info.default_factory() if info.default_factory else None
    if isinstance(d, BaseModel):
        if isinstance(d, PluginSpec):
            return f"default `{d.type}`"
        return "default: see keys"
    if d in ({}, [], None):
        return "optional"
    if isinstance(d, bool):
        return f"default `{str(d).lower()}`"
    return f"default `{d!r}`".replace("'", "")


def _lines(model: type[BaseModel], prefix: str, depth: int,
           skip: frozenset[str] = frozenset()) -> list[str]:
    out = []
    pad = "  " * depth
    for name, info in model.model_fields.items():
        if (model, name) in SKIP_FIELDS or name in skip:
            continue
        tp = info.annotation
        desc = DESCRIPTIONS[model][name]
        key = f"{prefix}{name}"
        expand = EXPANDS.get((model, name))
        sub_skip = frozenset({"total_usd"}) if (model, name) == (ArmDoc, "budget") else frozenset()
        if expand is not None:
            kind, sub = expand
            keys = [k for k in sub.model_fields if (sub, k) not in SKIP_FIELDS and k not in sub_skip]
            sh = "{" + ", ".join(keys) + "}"
            if kind == "by_name":
                sh = "{NAME: " + sh + "}"
        else:
            sh = shape(tp)
        out.append(f"{pad}- `{key}`: `{sh}` ({_default(info)}): {desc}")
        nested: type[BaseModel] | None = None
        child = f"{key}."
        if expand is not None:
            nested = expand[1]
            if expand[0] == "by_name":
                child = f"{key}.NAME."
        elif isinstance(tp, type) and issubclass(tp, BaseModel) and not _is_plugin(tp):
            nested = tp
        else:
            inner = [a for a in typing.get_args(tp)
                     if isinstance(a, type) and issubclass(a, BaseModel) and not _is_plugin(a)]
            if inner:
                nested = inner[-1]
                if typing.get_origin(tp) is dict:
                    child = f"{key}.NAME."
                elif typing.get_origin(tp) is list:
                    child = f"{key}[]."
        if nested is not None:
            out += _lines(nested, child, depth + 1, sub_skip)
    return out


def spec_reference() -> str:
    """The Markdown body of README.md's "Spec reference" section."""
    lines = [
        ("Every key of an experiment YAML, generated from the pydantic models "
         "(`swarmlab spec-reference` prints this list). Unknown keys are errors. Wherever the "
         f"shape is `{PLUGIN_SHAPE}` the value is {PLUGIN_NOTE}."),
        "",
        *_lines(ExperimentDoc, "", 0),
        "",
        "Example: a gossip arm where each agent reaches one partner per round, with its own caps:",
        "",
        "```yaml",
        "arms:",
        "  gossip:",
        "    world: flaggame",
        "    participants:",
        "      - {type: llm, count: 6, params: {model: \"anthropic:claude-haiku-4-5\"}}",
        "    medium: {topology: {type: gossip, params: {k: 1}}}",
        "    budget: {soft_usd: 0, hard_usd: 0.5}",
        "```",
    ]
    return "\n".join(lines)


# ---- readable validation errors --------------------------------------------------------------
def _step(tp: Any, key: Any) -> Any:
    """The annotation one level below `tp` along the path element `key` (None when unknown)."""
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin in (typing.Union, types.UnionType):
        for a in args:
            nxt = _step(a, key)
            if nxt is not None:
                return nxt
        return None
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        info = tp.model_fields.get(str(key))
        if info is None:
            return None
        expand = EXPANDS.get((tp, str(key)))
        if expand is not None:
            return expand[1] if expand[0] == "mapping" else dict[str, expand[1]]
        return info.annotation
    if origin is list and isinstance(key, int):
        return args[0] if args else None
    if origin is dict and isinstance(key, str):
        return args[1] if len(args) > 1 else None
    return None


def _expected(tp: Any) -> str | None:
    if _is_plugin(tp):
        return f"{PLUGIN_SHAPE} or a bare NAME string"
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        keys = [k for k in tp.model_fields if (tp, k) not in SKIP_FIELDS]
        extra = ""
        if tp is MediumSpec:
            extra = (f"; topology is {PLUGIN_SHAPE}, e.g. "
                     "`medium: {topology: {type: gossip, params: {k: 1}}}`")
        return "a mapping with keys " + ", ".join(keys) + extra
    origin = typing.get_origin(tp)
    if origin is list:
        inner = typing.get_args(tp)[0]
        return f"a list of {shape(inner)}" + (" (or bare NAME strings)" if _is_plugin(inner) else "")
    if origin is dict:
        return f"a mapping {shape(tp)}"
    return None


def explain_validation_error(err: ValidationError, root: type[BaseModel] = ExperimentDoc) -> str:
    """One line per error: the key path, what is wrong, and the shape expected there."""
    lines = []
    for e in err.errors():
        loc = list(e.get("loc", ()))
        path = ".".join(str(x) for x in loc) or "(document)"
        # the deepest documented container along the path
        tp: Any = root
        container: Any = root
        cpath: list[Any] = []
        for i, key in enumerate(loc):
            nxt = _step(tp, key)
            if nxt is None:
                break
            tp = nxt
            if _expected(tp) is not None:
                container, cpath = tp, loc[: i + 1]
        if e.get("type") == "extra_forbidden":
            what = f"unknown key {loc[-1]!r}" if loc else "unknown key"
            if cpath == loc:  # the extra key itself named a field (cannot happen) - be safe
                cpath = loc[:-1]
        else:
            what = e.get("msg", "invalid")
        where = ".".join(str(x) for x in cpath) or "the document"
        exp = _expected(container)
        line = f"{path}: {what}"
        if exp:
            line += f"; {where} expects {exp}"
        lines.append(line)
    return "\n".join(lines)


def readme_section(readme_text: str) -> str | None:
    """The generated block between the markers in README text, or None if absent."""
    if START not in readme_text or END not in readme_text:
        return None
    return readme_text.split(START, 1)[1].split(END, 1)[0].strip("\n")


def replace_readme_section(readme_text: str) -> str:
    """README text with the generated block replaced by the current reference."""
    head, rest = readme_text.split(START, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{START}\n{spec_reference()}\n{END}{tail}"
