"""The public API: `Experiment` and `Run` (docs/INTERFACE.md §0).

Decisions where the contract is silent:

- `Experiment` has two extra optional fields: `arm` (set by `from_yaml`/`from_spec`; None for a
  Python-defined experiment, which gives run id `<name>__s<seed>`) and `options` (run-option
  defaults from YAML, e.g. `max_rounds`, `commit`). `run(seed, max_rounds=None, ...)` resolves
  each option as: explicit argument, else `options`, else the `RunOptions` default; `max_rounds`
  must come from one of the first two.
- `medium` defaults to a fresh `Board()` per experiment (`default_factory`); the runner deep-copies
  it anyway.
- `to_spec`: participants are one `PluginSpec` per prototype (= per agent); metric strings become
  `PluginSpec(type=name)`, `Metric` objects their `spec()`. A metric built with `Metric.from_fn`
  has no importable type, so it runs but does not survive `from_spec`.
- `from_spec` resolves plugins through entry points (`swarmlab.worlds`, `swarmlab.participants`,
  `swarmlab.metrics`) or `module:Class`; the medium is rebuilt with `Board(**MediumSpec)` (the
  board resolves its own topology and policies). Metrics come back as `Metric` objects.
- `to_yaml(path)` writes a one-arm experiment document (arm name = `arm` or "default") whose
  top-level `options` are exactly `self.options`; `from_yaml(path, arm)` reads it back.
- `Run` reads everything from the run directory. `Run.events` is `logical_view(...)` (plain dicts
  without `seq`/`ts`); `Run.events_all` yields every typed `Event`. `Run.metrics` is read from the
  logged `metric` events. `Run.fork(at_round, experiment=None)` returns a `ForkHandle`; call
  `.run(out=None)` on it to execute the fork (`run.fork(4).run()`). `Run.resume()` resumes the
  same directory and returns a fresh `Run`. `Run.load(dir)` replays the log (raising
  `ReplayMismatch` on divergence) and returns the `Run`.
- M1b: `providers: dict[str, Provider] | None` overrides the preset provider for a model prefix
  (e.g. `{"vllm": OpenAICompatProvider("vllm", base_url=...)}` or a `FakeProvider` with test
  pricing); it round-trips through `RunSpec.providers` / YAML `providers:`. At construction every
  participant's model (`participant_model(p)`: a `model` attribute or `params["model"]` of the form
  `"<prefix>:<id>"`) is resolved and priced, so an unknown prefix is a `ValueError` and an unpriced
  model an `UnknownModelPricing` there, not at call time. The resolved providers are cached on the
  experiment (`resolved_providers()`) and shared by its runs, so a provider's `calls` counter
  sees every run of the experiment. Providers are never deep-copied or pickled.
- `estimate(seed, max_rounds, calls_per_turn=2, prompt_tokens=3000, completion_tokens=300)`:
  `agents * max_rounds * calls_per_turn` calls per model-backed participant, each priced at
  `prompt_tokens * p_in + completion_tokens * p_out`; participants without a model cost 0. Returns
  `{"arm", "agents", "llm_agents", "rounds", "calls", "usd", "by_model", "budget"}`. `seed` is
  accepted for symmetry with `run` (the agent count does not depend on it).
- M1b (WP7): `probes` holds `Probe` objects; strings and `{type, params}` mappings are built
  through the `swarmlab.probes` entry points at construction, and a probe's `coder_model()` is
  priced there like participant models. `estimate` caps `calls_per_turn` at a participant's own
  `max_calls` when it has one, and also counts one probe call per probed agent per
  probed round (`probe_calls`, priced like a turn call, `measurement_usd`, included in `usd`).
  `Run.probes` reads the `probe` events: `{name: [(round, agent, parsed, ok), ...]}`.
- `Run.spend` reads `run.json["ledger"]`: `{"swarm", "measurement", "reserved", "calls"}`.
  `Run.resume(budget=None)` passes the budget through to the runner (see `Runner.resume`).
- When a `Run` came from `Experiment.run`, `fork`/`resume` reuse that in-memory experiment
  (so plugins defined inline in a script work); a loaded `Run` rebuilds it with `from_spec`.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from .events import Event, EventLog, logical_view
from .ids import run_id as make_run_id
from .medium.board import Board
from .metrics.base import Metric
from .participants.base import Participant
from .probes import Probe, build_probe
from .providers import resolve as resolve_provider
from .providers.base import Provider, split_model
from .registry import build
from .runner import Runner
from .spec import (
    Budget,
    MediumSpec,
    PluginSpec,
    RunOptions,
    RunSpec,
    arm_to_runspec,
    dump_experiment_yaml,
    load_experiment_yaml,
    runspec_to_doc,
)
from .world.base import World


class Experiment(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    world: World
    participants: list[Participant]
    medium: Board = Field(default_factory=Board)
    metrics: list[str | Metric] = []
    budget: Budget = Field(default_factory=Budget)
    probes: list = []
    interventions: list = []
    arm: str | None = None
    options: dict = {}
    providers: dict[str, Provider] | None = None
    _resolved: dict[str, Provider] = PrivateAttr(default_factory=dict)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.probes = [p if isinstance(p, Probe) else build_probe(p) for p in self.probes]
        names = [p.name for p in self.probes]
        if len(set(names)) != len(names):
            raise ValueError(f"probe names must be unique, got {names}")
        # after validation, so UnknownModelPricing propagates as itself (not a ValidationError)
        models = [participant_model(p) for p in self.participants]
        models += [p.coder_model() for p in self.probes]
        for model in models:
            if model is not None:
                provider, mid = self.provider_for(model)
                provider.model_pricing(mid)

    def provider_for(self, model: str) -> tuple[Provider, str]:
        """(provider, model id) for `"<prefix>:<id>"`: the override, else a cached preset."""
        prefix, mid = split_model(model)
        if prefix not in self._resolved:
            self._resolved[prefix] = resolve_provider(model, self.providers)[0]
        return self._resolved[prefix], mid

    def resolved_providers(self) -> dict[str, Provider]:
        """Providers by prefix: the overrides plus every preset resolved so far."""
        return {**self._resolved, **(self.providers or {})}

    def estimate(self, seed: int, max_rounds: int, calls_per_turn: int = 2,
                 prompt_tokens: int = 3000, completion_tokens: int = 300) -> dict:
        by_model: dict[str, float] = {}
        llm = 0
        calls = 0
        probe_calls = 0
        measurement = 0.0
        for p in self.participants:
            model = participant_model(p)
            if model is None:
                continue
            llm += 1
            provider, mid = self.provider_for(model)
            p_in, p_out, _ = provider.model_pricing(mid)
            per_call = (prompt_tokens * p_in + completion_tokens * p_out) / 1e6
            own = getattr(p, "max_calls", None)
            cpt = min(calls_per_turn, own) if isinstance(own, int) and own > 0 else calls_per_turn
            calls += cpt * max_rounds
            by_model[model] = by_model.get(model, 0.0) + per_call * cpt * max_rounds
            if callable(getattr(p, "probe_context", None)):
                for probe in self.probes:
                    n = max_rounds // max(1, probe.every)
                    probe_calls += n
                    measurement += per_call * n
        return {
            "arm": self.arm, "agents": len(self.participants), "llm_agents": llm,
            "rounds": max_rounds, "calls": calls,
            "usd": sum(by_model.values()) + measurement, "by_model": by_model,
            "probe_calls": probe_calls, "measurement_usd": measurement,
            "budget": self.budget.model_dump(mode="json"),
        }

    # ---- running -----------------------------------------------------------------------------
    def _options(self, seed: int, max_rounds: int | None, **kw: Any) -> RunOptions:
        merged = {k: v for k, v in self.options.items() if k != "seed"}
        merged.update({k: v for k, v in kw.items() if v is not None})
        if max_rounds is not None:
            merged["max_rounds"] = max_rounds
        if "max_rounds" not in merged:
            raise ValueError("max_rounds is required (argument or experiment options)")
        return RunOptions(seed=seed, **merged)

    def run(
        self,
        seed: int,
        max_rounds: int | None = None,
        *,
        commit: Literal["round_end", "immediate"] | None = None,
        max_calls_per_turn: int | None = None,
        snapshot_every: int | None = None,
        out: Path | str = "runs",
        concurrency: int | None = None,
    ) -> Run:
        options = self._options(seed, max_rounds, commit=commit, max_calls_per_turn=max_calls_per_turn,
                                snapshot_every=snapshot_every, concurrency=concurrency)
        run_dir = Path(out) / make_run_id(self.name, self.arm, seed)
        Runner(run_dir, self, options).live()
        return Run(run_dir, experiment=self)

    # ---- serialisation -----------------------------------------------------------------------
    def to_spec(self, seed: int, max_rounds: int, **run_options: Any) -> RunSpec:
        metrics = [
            PluginSpec(type=m) if isinstance(m, str) else PluginSpec(**m.spec()) for m in self.metrics
        ]
        return RunSpec(
            experiment=self.name,
            arm=self.arm,
            world=PluginSpec(**self.world.spec()),
            participants=[PluginSpec(**p.spec()) for p in self.participants],
            medium=MediumSpec(**self.medium.spec()["params"]),
            metrics=metrics,
            budget=self.budget,
            options=RunOptions(seed=seed, max_rounds=max_rounds, **run_options),
            providers={k: PluginSpec(**p.spec()) for k, p in (self.providers or {}).items()},
            probes=[PluginSpec(**p.spec()) for p in self.probes],
        )

    @classmethod
    def from_spec(cls, spec: RunSpec) -> Experiment:
        medium = spec.medium.model_dump(mode="json")
        return cls(
            name=spec.experiment,
            arm=spec.arm,
            world=build(spec.world, "swarmlab.worlds"),
            participants=[build(p, "swarmlab.participants") for p in spec.participants],
            medium=Board(**medium),
            metrics=[build(m, "swarmlab.metrics") for m in spec.metrics],
            budget=spec.budget,
            providers={k: build(p, "swarmlab.providers") for k, p in spec.providers.items()} or None,
            probes=[build(p, "swarmlab.probes") for p in spec.probes],
        )

    @classmethod
    def from_yaml(cls, path: Path | str, arm: str) -> Experiment:
        doc = load_experiment_yaml(path)
        if arm not in doc["arms"]:
            from .spec import SpecError

            raise SpecError(f"unknown arm {arm!r}; available: {sorted(doc['arms'])}")
        options = {**doc["options"], **doc["arms"][arm]["options"]}
        options.pop("seed", None)
        spec = arm_to_runspec(doc, arm, seed=0, max_rounds=options.get("max_rounds", 1))
        exp = cls.from_spec(spec)
        exp.options = options
        return exp

    def to_yaml(self, path: Path | str) -> None:
        spec = self.to_spec(0, self.options.get("max_rounds", 1))
        doc = runspec_to_doc(spec, self.arm or "default")
        doc["options"] = {k: v for k, v in self.options.items() if k != "seed"}
        dump_experiment_yaml(doc, path)


class ForkHandle:
    """`run.fork(at_round, experiment)` result; `.run(out=None)` executes the fork."""

    def __init__(self, parent: Run, at_round: int, experiment: Experiment | None) -> None:
        self.parent = parent
        self.at_round = at_round
        self.experiment = experiment

    def run(self, out: Path | str | None = None, *, max_rounds: int | None = None) -> Run:
        parent_exp = self.parent.experiment
        runner = Runner(self.parent.dir, parent_exp, self.parent.spec.options)
        child = runner.fork(self.at_round, self.experiment or parent_exp, out, max_rounds=max_rounds)
        return Run(child.dir, experiment=self.experiment or parent_exp)


class Run:
    def __init__(self, dir: Path | str, *, experiment: Experiment | None = None) -> None:
        self.dir = Path(dir)
        self._experiment = experiment
        self._meta = Runner._read_meta_at(self.dir)

    # ---- metadata ----------------------------------------------------------------------------
    @property
    def meta(self) -> dict:
        return self._meta

    @property
    def spec(self) -> RunSpec:
        return RunSpec.model_validate(self._meta["spec"])

    @property
    def id(self) -> str:
        return self._meta["run_id"]

    @property
    def score(self) -> dict:
        return self._meta["score"]

    @property
    def status(self) -> str:
        return self._meta["status"]

    @property
    def end_reason(self) -> str | None:
        return self._meta["end_reason"]

    @property
    def spend(self) -> dict:
        led = self._meta.get("ledger") or {}
        return {k: led.get(k, 0) for k in ("swarm", "measurement", "reserved", "calls")}

    @property
    def experiment(self) -> Experiment:
        if self._experiment is None:
            self._experiment = Experiment.from_spec(self.spec)
        return self._experiment

    # ---- events ------------------------------------------------------------------------------
    @property
    def events_all(self) -> Iterator[Event]:
        log = EventLog(self.dir / "events.jsonl")
        yield from log

    @property
    def events(self) -> Iterator[dict]:
        return logical_view(self.events_all)

    @property
    def metrics(self) -> dict[str, list[tuple[int, float | None, int]]]:
        out: dict[str, list[tuple[int, float | None, int]]] = {}
        for ev in self.events_all:
            if ev.type == "metric":
                out.setdefault(ev.name, []).append((ev.round, ev.value, ev.denominator))
        return out

    @property
    def probes(self) -> dict[str, list[tuple[int, str, dict, bool]]]:
        """Probe answers by probe name: `(round, agent, parsed, ok)` in log order."""
        out: dict[str, list[tuple[int, str, dict, bool]]] = {}
        for ev in self.events_all:
            if ev.type == "probe":
                out.setdefault(ev.probe, []).append((ev.round, str(ev.agent), dict(ev.parsed), ev.ok))
        return out

    # ---- operations --------------------------------------------------------------------------
    def view(self) -> Path:
        from .viewer.build import build as build_view

        return build_view(self.dir)

    def fork(self, at_round: int, experiment: Experiment | None = None) -> ForkHandle:
        return ForkHandle(self, at_round, experiment)

    def resume(self, budget: Budget | None = None) -> Run:
        Runner(self.dir, self.experiment, self.spec.options).resume(budget=budget)
        return Run(self.dir, experiment=self._experiment)

    def replay(self) -> dict:
        return Runner(self.dir, self.experiment, self.spec.options).replay()

    @classmethod
    def load(cls, dir: Path | str) -> Run:
        run = cls(dir)
        run.replay()
        return cls(dir, experiment=run._experiment)

    def __repr__(self) -> str:
        return f"Run({self.id!r}, status={self.status!r}, dir={str(self.dir)!r})"


def participant_model(p: Any) -> str | None:
    """The `"<prefix>:<id>"` model a participant calls, or None (scripted participants)."""
    model = getattr(p, "model", None)
    if not isinstance(model, str):
        model = (getattr(p, "params", None) or {}).get("model")
    return model if isinstance(model, str) and ":" in model else None
