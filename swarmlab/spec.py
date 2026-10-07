"""Resolved run specification, hashing, and YAML helpers (docs/INTERFACE.md §3).

Decisions where the contract is silent:

- `spec_hash` is SHA-256 over `json.dumps(spec.model_dump(mode="json"), sort_keys=True,
  separators=(",", ":"), ensure_ascii=True)`. Key order in params never changes the hash; list
  order (agents, metrics, policies) does, because it is semantic.
- Experiment YAML shape (validated by `load_experiment_yaml`; unknown keys are errors)::

      name: flag-gossip
      budget: {soft_usd: 0, hard_usd: 0, measurement_usd: 0}   # optional
      providers:                                   # optional (M1b): overrides by model prefix
        vllm: {type: openai_compat, params: {name: vllm, base_url: "http://host:8000/v1"}}
      options: {max_rounds: 20, commit: round_end}  # optional; any RunOptions field
      seeds: [1, 2, 3]                             # optional; seeds `swarmlab run` runs per arm
      arms:
        A:
          world: {type: flaggame, params: {n_candidates: 8}}   # or the bare string "flaggame"
          participants:
            - {type: evidence_aggregator, count: 16, params: {}}   # count defaults to 1
          medium: {topology: gossip}       # optional; MediumSpec; topology/policies may be strings
          metrics: [belief.consensus, {type: belief.polarization, params: {threshold: 0.2}}]
          options: {commit: immediate}     # optional, merged over top-level options
          budget: {hard_usd: 5}            # optional, merged over top-level budget

  Anywhere a plugin is expected, a bare string `"x"` means `{type: "x", params: {}}`.
  Per-arm `options`/`budget` exist so commit policy can be an arm (DESIGN.md M2).
- `load_experiment_yaml` returns the *normalised* document as plain data (strings expanded to
  `{type, params}`, defaults filled), so `arm_to_runspec` and `dump_experiment_yaml` accept it.
- `git_identity` treats untracked files as clean (`git status --porcelain --untracked-files=no`),
  so run output written inside the repo does not mark later runs dirty.
- `seeds` (additive) is the list of seeds `swarmlab run SPEC` (no `--seed`) and
  `experiment_seeds(doc)` use; absent or empty means `[0]`. It is not part of any `RunSpec`, so
  it never changes a `spec_hash`. The normalised document omits it when empty.
- Validation failures raise `SpecError` (a `ValueError`); the CLI maps it to exit code 2.
- M3a: `interventions: [{type, params}]` (bare strings allowed) on the experiment and/or an arm;
  the run spec gets the experiment's list followed by the arm's. `RunSpec.interventions` is left
  out of `spec_hash` when empty, and empty lists are left out of the normalised document, so
  earlier specs, documents and hashes are unchanged.
- M3b: `MediumSpec.registry: bool = False` turns on the registry tools and
  `MediumSpec.claim_policy` (default `advisory`; `enforced`; bare strings allowed) links it to
  the world (swarmlab/medium/registry.py). `MediumSpec` serialises without either field while it
  is at its default, so earlier run specs, documents and hashes are unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    ValidationError,
    field_validator,
    model_serializer,
)

from ._io import atomic_write_bytes


class SpecError(ValueError):
    """The experiment YAML or run spec does not match the documented shape."""


class Budget(BaseModel):
    """Enforced from M1b (swarmlab/budget.py); a field <= 0 is not enforced."""

    model_config = ConfigDict(extra="forbid")
    soft_usd: float = 0.0
    hard_usd: float = 0.0
    measurement_usd: float = 0.0


class PluginSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str
    params: dict = {}


def _coerce_plugin(v: Any) -> Any:
    """A bare string `"x"` is shorthand for `{type: "x", params: {}}`."""
    return {"type": v, "params": {}} if isinstance(v, str) else v


class MediumSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topology: PluginSpec = PluginSpec(type="broadcast")
    delivery: Literal["pull", "push"] = "pull"
    push_limit: int = 20
    policies: list[PluginSpec] = []
    channels: list[str] = ["main"]
    registry: bool = False  # M3b
    claim_policy: PluginSpec = PluginSpec(type="advisory")  # M3b

    @field_validator("topology", "claim_policy", mode="before")
    @classmethod
    def _topology(cls, v: Any) -> Any:
        return _coerce_plugin(v)

    @model_serializer(mode="wrap")
    def _drop_m3b_defaults(self, handler: Any) -> Any:
        data = handler(self)
        if isinstance(data, dict):
            if data.get("registry") is False:
                data.pop("registry")
            if data.get("claim_policy") in ({"type": "advisory", "params": {}},
                                            PluginSpec(type="advisory")):
                data.pop("claim_policy")
        return data

    @field_validator("policies", mode="before")
    @classmethod
    def _policies(cls, v: Any) -> Any:
        return [_coerce_plugin(x) for x in v] if isinstance(v, (list, tuple)) else v


class RunOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int
    max_rounds: int
    commit: Literal["round_end", "immediate"] = "round_end"
    max_calls_per_turn: int = 20
    snapshot_every: int = 1
    concurrency: int = 32
    # M3a §2 paired runs: repeat i > 0 re-derives the agent streams (see runner docstring).
    # Omitted from dumps when 0, so ordinary runs keep their spec and spec hash.
    repeat: int = Field(default=0, ge=0)

    @model_serializer(mode="wrap")
    def _drop_default_repeat(self, handler: SerializerFunctionWrapHandler) -> dict:
        data = handler(self)
        if data.get("repeat") == 0:
            data.pop("repeat")
        return data


class RunSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    experiment: str
    arm: str | None = None
    world: PluginSpec
    participants: list[PluginSpec]  # one entry per agent, in agent order
    medium: MediumSpec
    metrics: list[PluginSpec]
    budget: Budget
    options: RunOptions
    providers: dict[str, PluginSpec] = {}  # M1b: provider overrides by model prefix
    probes: list[PluginSpec] = []  # M1b: probes run after each commit (swarmlab/probes.py)
    interventions: list[PluginSpec] = []  # M3a: swarmlab/interventions.py


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def spec_hash(run_spec: RunSpec) -> str:
    """SHA-256 hex of the canonical JSON of the resolved spec."""
    data = run_spec.model_dump(mode="json")
    if not data.get("probes"):  # M1b field; omitted when empty so earlier specs keep their hash
        data.pop("probes", None)
    if not data.get("interventions"):  # M3a, likewise
        data.pop("interventions", None)
    return hashlib.sha256(canonical_json(data).encode()).hexdigest()


# ---- experiment YAML --------------------------------------------------------------------------


class ParticipantGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: str
    count: int = Field(default=1, ge=1)
    params: dict = {}


class ArmDoc(BaseModel):
    model_config = ConfigDict(extra="forbid")
    world: PluginSpec
    participants: list[ParticipantGroup] = Field(min_length=1)
    medium: MediumSpec = MediumSpec()
    metrics: list[PluginSpec] = []
    probes: list[PluginSpec] = []
    interventions: list[PluginSpec] = []
    options: dict = {}
    budget: dict = {}

    @field_validator("world", mode="before")
    @classmethod
    def _world(cls, v: Any) -> Any:
        return _coerce_plugin(v)

    @field_validator("participants", mode="before")
    @classmethod
    def _participants(cls, v: Any) -> Any:
        return [_coerce_plugin(x) for x in v] if isinstance(v, (list, tuple)) else v

    @field_validator("metrics", "probes", "interventions", mode="before")
    @classmethod
    def _metrics(cls, v: Any) -> Any:
        return [_coerce_plugin(x) for x in v] if isinstance(v, (list, tuple)) else v


class ExperimentDoc(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    arms: dict[str, ArmDoc] = Field(min_length=1)
    budget: Budget = Budget()
    options: dict = {}
    providers: dict[str, PluginSpec] = {}
    seeds: list[int] = []
    interventions: list[PluginSpec] = []

    @field_validator("interventions", mode="before")
    @classmethod
    def _interventions(cls, v: Any) -> Any:
        return [_coerce_plugin(x) for x in v] if isinstance(v, (list, tuple)) else v

    @field_validator("providers", mode="before")
    @classmethod
    def _providers(cls, v: Any) -> Any:
        return {k: _coerce_plugin(x) for k, x in v.items()} if isinstance(v, dict) else v


_OPTION_FIELDS = set(RunOptions.model_fields)
_BUDGET_FIELDS = set(Budget.model_fields)


def validate_experiment_doc(doc: dict) -> dict:
    """Validate and normalise an experiment document (already parsed from YAML)."""
    if not isinstance(doc, dict):
        raise SpecError(f"experiment document must be a mapping, got {type(doc).__name__}")
    try:
        parsed = ExperimentDoc.model_validate(doc)
    except ValidationError as e:
        raise SpecError(str(e)) from e
    where = [("options", parsed.options, _OPTION_FIELDS)]
    for arm_name, arm in parsed.arms.items():
        where += [
            (f"arms.{arm_name}.options", arm.options, _OPTION_FIELDS),
            (f"arms.{arm_name}.budget", arm.budget, _BUDGET_FIELDS),
        ]
    for label, mapping, allowed in where:
        unknown = set(mapping) - allowed
        if unknown:
            raise SpecError(f"{label}: unknown keys {sorted(unknown)}; allowed {sorted(allowed)}")
    if len(set(parsed.seeds)) != len(parsed.seeds):
        raise SpecError(f"seeds: duplicates in {parsed.seeds}")
    out = parsed.model_dump(mode="json")
    if not out["seeds"]:
        out.pop("seeds")
    if not out["interventions"]:
        out.pop("interventions")
    for arm in out["arms"].values():
        if not arm["interventions"]:
            arm.pop("interventions")
    return out


def experiment_seeds(doc: dict) -> list[int]:
    """The document's `seeds`, or `[0]` when it lists none."""
    return list(doc.get("seeds") or [0])


def load_experiment_yaml(path: Path | str) -> dict:
    """Read and validate an experiment YAML file; returns the normalised document."""
    try:
        raw = yaml.safe_load(Path(path).read_text())
    except yaml.YAMLError as e:
        raise SpecError(f"{path}: invalid YAML: {e}") from e
    return validate_experiment_doc(raw)


def dump_experiment_yaml(doc: dict, path: Path | str) -> Path:
    """Validate, normalise, and write an experiment document as YAML."""
    data = validate_experiment_doc(doc)
    text = yaml.safe_dump(data, sort_keys=False, default_flow_style=False, allow_unicode=True)
    return atomic_write_bytes(path, text.encode("utf-8"))


def arm_to_runspec(doc: dict, arm: str, seed: int, **option_overrides: Any) -> RunSpec:
    """Resolve one arm of an experiment document into a hashable RunSpec.

    Options precedence (lowest to highest): RunOptions defaults, top-level `options`, the arm's
    `options`, `seed`, then `option_overrides` (overrides whose value is None are ignored, so CLI
    flags can be passed through unconditionally). `participants[].count` expands into one
    PluginSpec per agent, in document order.
    """
    norm = validate_experiment_doc(doc)
    if arm not in norm["arms"]:
        raise SpecError(f"unknown arm {arm!r}; available: {sorted(norm['arms'])}")
    a = norm["arms"][arm]
    overrides = {k: v for k, v in option_overrides.items() if v is not None}
    unknown = set(overrides) - _OPTION_FIELDS
    if unknown:
        raise SpecError(f"unknown run options {sorted(unknown)}")
    options = {**norm["options"], **a["options"], "seed": seed, **overrides}
    participants = [
        PluginSpec(type=g["type"], params=g["params"])
        for g in a["participants"]
        for _ in range(g["count"])
    ]
    try:
        return RunSpec(
            experiment=norm["name"],
            arm=arm,
            world=PluginSpec(**a["world"]),
            participants=participants,
            medium=MediumSpec(**a["medium"]),
            metrics=[PluginSpec(**m) for m in a["metrics"]],
            probes=[PluginSpec(**p) for p in a["probes"]],
            interventions=[PluginSpec(**i) for i in norm.get("interventions", []) + a.get("interventions", [])],
            budget=Budget(**{**norm["budget"], **a["budget"]}),
            options=RunOptions(**options),
            providers={k: PluginSpec(**v) for k, v in norm["providers"].items()},
        )
    except ValidationError as e:
        raise SpecError(str(e)) from e


def runspec_to_doc(run_spec: RunSpec, arm: str | None = None) -> dict:
    """Inverse of `arm_to_runspec`: a one-arm experiment document.

    Consecutive identical participant entries collapse into one group with a `count`. The
    seed is dropped from `options` (it is supplied at run time).
    """
    groups: list[dict] = []
    for p in run_spec.participants:
        if groups and groups[-1]["type"] == p.type and groups[-1]["params"] == p.params:
            groups[-1]["count"] += 1
        else:
            groups.append({"type": p.type, "count": 1, "params": p.params})
    options = run_spec.options.model_dump(mode="json")
    options.pop("seed")
    arm_name = arm or run_spec.arm or "default"
    return validate_experiment_doc(
        {
            "name": run_spec.experiment,
            "budget": run_spec.budget.model_dump(mode="json"),
            "options": options,
            "providers": {k: v.model_dump(mode="json") for k, v in run_spec.providers.items()},
            "arms": {
                arm_name: {
                    "world": run_spec.world.model_dump(mode="json"),
                    "participants": groups,
                    "medium": run_spec.medium.model_dump(mode="json"),
                    "metrics": [m.model_dump(mode="json") for m in run_spec.metrics],
                    "probes": [p.model_dump(mode="json") for p in run_spec.probes],
                    "interventions": [i.model_dump(mode="json") for i in run_spec.interventions],
                }
            },
        }
    )


def dump_runspec_yaml(run_spec: RunSpec, path: Path | str) -> Path:
    """Archive a resolved RunSpec as YAML (e.g. `artifacts/spec.yaml`)."""
    data = run_spec.model_dump(mode="json")
    text = yaml.safe_dump(data, sort_keys=False, default_flow_style=False, allow_unicode=True)
    return atomic_write_bytes(path, text.encode("utf-8"))


def load_runspec_yaml(path: Path | str) -> RunSpec:
    """Read a RunSpec written by `dump_runspec_yaml`."""
    try:
        return RunSpec.model_validate(yaml.safe_load(Path(path).read_text()))
    except (yaml.YAMLError, ValidationError) as e:
        raise SpecError(f"{path}: {e}") from e


def git_identity(repo_dir: Path | str = ".") -> tuple[str, bool]:
    """(HEAD commit sha, dirty) for the git work tree containing `repo_dir`.

    `SWARMLAB_GIT_COMMIT`, when set, wins (an installed wheel, e.g. inside an HF Job: the
    launcher states the commit it built from; `SWARMLAB_GIT_DIRTY=1` marks a dirty build).
    Returns ("unknown", True) when `repo_dir` is not inside a git repository or git is missing.
    Dirty means tracked files differ from HEAD (untracked files are ignored).
    """
    env_commit = os.environ.get("SWARMLAB_GIT_COMMIT", "").strip()
    if env_commit:
        return env_commit, os.environ.get("SWARMLAB_GIT_DIRTY", "1").strip() not in ("0", "")

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo_dir), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout

    try:
        commit = git("rev-parse", "HEAD").strip()
        dirty = bool(git("status", "--porcelain", "--untracked-files=no").strip())
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True
    return commit, dirty
