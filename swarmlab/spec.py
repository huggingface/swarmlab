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
- Validation failures raise `SpecError` (a `ValueError`); the CLI maps it to exit code 2.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

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

    @field_validator("topology", mode="before")
    @classmethod
    def _topology(cls, v: Any) -> Any:
        return _coerce_plugin(v)

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


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def spec_hash(run_spec: RunSpec) -> str:
    """SHA-256 hex of the canonical JSON of the resolved spec."""
    data = run_spec.model_dump(mode="json")
    if not data.get("probes"):  # M1b field; omitted when empty so earlier specs keep their hash
        data.pop("probes", None)
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

    @field_validator("metrics", "probes", mode="before")
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
    return parsed.model_dump(mode="json")


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

    Returns ("unknown", True) when `repo_dir` is not inside a git repository or git is missing.
    Dirty means tracked files differ from HEAD (untracked files are ignored).
    """

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
