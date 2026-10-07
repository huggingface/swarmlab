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
- Catalog pricing: a model the provider cannot price is priced from the model catalog
  (`providers.ensure_pricing`; today the HF router listing for `hf:`). The price used is pinned
  into `providers[prefix].params["pricing"]` (the preset becomes an explicit override), so the run
  spec records it: replay, resume and fork never need the catalog again, and a catalog price
  change gives a different `spec_hash`.
- Running a whole experiment: `Experiment.arms_from_yaml(path)` builds every arm;
  `run_all(seeds, max_rounds=None, out="runs", skip_existing=True, ...)` runs one arm over seeds
  in order and returns the `Run`s. `existing(seed, max_rounds, out, ...)` tells what is already
  on disk at `out/<run_id>`: `("new", None)`, `("same", Run)` (same `spec_hash`; skipped) or
  `("different", Run)` (raises `FileExistsError` in `run_all`; pass a fresh `out`, or
  `rerun=True` to write `out/<run_id>__r<N>` with the smallest free N >= 2).
- `estimate(..., seeds=None)`: when `seeds` is given the result adds `runs = len(seeds)` and
  `total_usd = usd * runs` (the agent count does not depend on the seed).
- `estimate(..., prompt_growth=0)` models full-memory context growth: a call in round r
  (1-based) is priced with `prompt_tokens + prompt_growth * (r - 1)` prompt tokens (a probe
  after round r likewise). `usd`, `by_model` and `measurement_usd` use the growth-adjusted
  tokens; `usd_flat` is the same estimate with growth 0 (equal to `usd` when growth is 0), and
  `total_usd_flat` accompanies `total_usd` when `seeds` is given.
- `Run.summary()` is the CLI's run summary: run_dir, run_id, arm, seed, spec_hash, status,
  end_reason, last_round, score, metrics (last value per name), spend (+ parent_run/fork_round
  for a fork; + `self_hosted`: the model prefixes served self-hosted, e.g. `["vllm"]`, only when
  there are any, so a $0 spend reads as compute time, not free). `Run.load(dir, run_id=None)` accepts a run dir, or a parent dir plus run id.
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
from .providers import catalog_priced, ensure_pricing
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
    SpecError,
    arm_to_runspec,
    dump_experiment_yaml,
    load_experiment_yaml,
    runspec_to_doc,
    spec_hash,
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
                row = ensure_pricing(provider, split_model(model)[0], mid)
                if mid in catalog_priced(provider):
                    self._pin_price(split_model(model)[0], provider, mid, row)

    def _pin_price(self, prefix: str, provider: Provider, mid: str, row: tuple) -> None:
        """Record a catalog price in the provider's params so `to_spec` carries it."""
        params = getattr(provider, "params", None)
        if not isinstance(params, dict):
            return
        params["pricing"] = {**(params.get("pricing") or {}), mid: [float(x) for x in row]}
        if not self.providers or self.providers.get(prefix) is not provider:
            self.providers = {**(self.providers or {}), prefix: provider}

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
                 prompt_tokens: int = 3000, completion_tokens: int = 300,
                 seeds: list[int] | None = None, prompt_growth: int = 0) -> dict:
        rounds = range(1, max_rounds + 1)
        # sum over rounds of the prompt tokens of one call per round (flat and growth-adjusted)
        flat_tokens = prompt_tokens * max_rounds
        grown_tokens = sum(prompt_tokens + prompt_growth * (r - 1) for r in rounds)
        by_model: dict[str, float] = {}
        by_model_flat: dict[str, float] = {}
        llm = 0
        calls = 0
        probe_calls = 0
        measurement = 0.0
        measurement_flat = 0.0
        for p in self.participants:
            model = participant_model(p)
            if model is None:
                continue
            llm += 1
            provider, mid = self.provider_for(model)
            p_in, p_out, _ = provider.model_pricing(mid)
            own = getattr(p, "max_calls", None)
            cpt = min(calls_per_turn, own) if isinstance(own, int) and own > 0 else calls_per_turn
            calls += cpt * max_rounds
            out_usd = completion_tokens * p_out * max_rounds
            by_model[model] = by_model.get(model, 0.0) + cpt * (grown_tokens * p_in + out_usd) / 1e6
            by_model_flat[model] = (by_model_flat.get(model, 0.0)
                                    + cpt * (flat_tokens * p_in + out_usd) / 1e6)
            if callable(getattr(p, "probe_context", None)):
                for probe in self.probes:
                    every = max(1, probe.every)
                    probed = [r for r in rounds if r % every == 0]
                    probe_calls += len(probed)
                    tokens = sum(prompt_tokens + prompt_growth * (r - 1) for r in probed)
                    measurement += (tokens * p_in + completion_tokens * p_out * len(probed)) / 1e6
                    measurement_flat += (prompt_tokens * p_in
                                         + completion_tokens * p_out) * len(probed) / 1e6
        out = {
            "arm": self.arm, "agents": len(self.participants), "llm_agents": llm,
            "rounds": max_rounds, "calls": calls,
            "usd": sum(by_model.values()) + measurement, "by_model": by_model,
            "probe_calls": probe_calls, "measurement_usd": measurement,
            "budget": self.budget.model_dump(mode="json"),
            "prompt_tokens": prompt_tokens, "prompt_growth": prompt_growth,
            "usd_flat": sum(by_model_flat.values()) + measurement_flat,
        }
        if seeds is not None:
            out["runs"] = len(seeds)
            out["total_usd"] = out["usd"] * len(seeds)
            out["total_usd_flat"] = out["usd_flat"] * len(seeds)
        return out

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
        run_dir: Path | str | None = None,
    ) -> Run:
        """Run one seed into `out/<run_id>` (or exactly `run_dir` when given)."""
        options = self._options(seed, max_rounds, commit=commit, max_calls_per_turn=max_calls_per_turn,
                                snapshot_every=snapshot_every, concurrency=concurrency)
        target = Path(run_dir) if run_dir is not None else Path(out) / self.run_id(seed)
        Runner(target, self, options).live()
        return Run(target, experiment=self)

    def run_id(self, seed: int) -> str:
        return make_run_id(self.name, self.arm, seed)

    def spec_hash(self, seed: int, max_rounds: int | None = None, **run_options: Any) -> str:
        """`spec_hash` of the run `run(seed, max_rounds, **run_options)` would start."""
        return spec_hash(self.to_spec(**self._options(seed, max_rounds, **run_options).model_dump()))

    def existing(self, seed: int, max_rounds: int | None = None, out: Path | str = "runs",
                 **run_options: Any) -> tuple[Literal["new", "same", "different"], Run | None]:
        """What `out/<run_id>` already holds for this seed (see module doc)."""
        d = Path(out) / self.run_id(seed)
        if not (d / "run.json").exists():
            return "new", None
        run = Run(d)
        same = run.meta.get("spec_hash") == self.spec_hash(seed, max_rounds, **run_options)
        return ("same" if same else "different"), run

    @staticmethod
    def rerun_dir(out: Path | str, run_id: str) -> Path:
        n = 2
        while (Path(out) / f"{run_id}__r{n}").exists():
            n += 1
        return Path(out) / f"{run_id}__r{n}"

    def run_all(self, seeds: list[int], max_rounds: int | None = None, *, out: Path | str = "runs",
                skip_existing: bool = True, rerun: bool = False, **run_options: Any) -> list[Run]:
        """Run this arm once per seed, in order; returns the runs (skipped ones included)."""
        runs: list[Run] = []
        for seed in seeds:
            state, found = self.existing(seed, max_rounds, out, **run_options)
            if rerun and state != "new":
                runs.append(self.run(seed, max_rounds, out=out, **run_options,
                                     run_dir=self.rerun_dir(out, self.run_id(seed))))
            elif state == "same" and skip_existing and found is not None:
                runs.append(found)
            elif state == "different" and found is not None:
                raise FileExistsError(
                    f"{found.dir} holds a run with a different spec_hash; pass rerun=True "
                    "(CLI: --rerun) or another out dir")
            else:
                runs.append(self.run(seed, max_rounds, out=out, **run_options))
        return runs

    @classmethod
    def arms_from_yaml(cls, path: Path | str) -> dict[str, Experiment]:
        """Every arm of an experiment YAML, in document order."""
        return {arm: cls.from_yaml(path, arm) for arm in load_experiment_yaml(path)["arms"]}

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

    # ---- M4: publish (docs/INTERFACE-M4.md §2) ------------------------------------------------
    def publish(self, runs_dir: Path | str = "runs", *, repo: str | None = None,
                public: bool = False, tag: list[str] | str | None = None, api: Any = None) -> dict:
        """Publish this experiment's finished runs under `runs_dir` (see `swarmlab.publish`)."""
        from .publish import publish

        return publish(runs_dir, repo, public=public, tag=tag, api=api, experiment=self.name)


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
    def load(cls, dir: Path | str, run_id: str | None = None) -> Run:
        """Replay a run from disk: `Run.load("runs/<id>")` or `Run.load("runs", "<id>")`."""
        path = Path(dir) / run_id if run_id is not None else Path(dir)
        run = cls(path)
        run.replay()
        return cls(path, experiment=run._experiment)

    def summary(self) -> dict[str, Any]:
        meta = self.meta
        metrics = {
            name: {"value": vals[-1][1], "denominator": vals[-1][2], "round": vals[-1][0]}
            for name, vals in self.metrics.items()
            if vals
        }
        out = {
            "run_dir": str(self.dir),
            "run_id": self.id,
            "arm": meta.get("arm"),
            "seed": (meta.get("spec") or {}).get("options", {}).get("seed"),
            "spec_hash": meta.get("spec_hash"),
            "status": self.status,
            "end_reason": self.end_reason,
            "last_round": meta.get("last_round"),
            "score": self.score,
            "metrics": metrics,
            "spend": self.spend,
        }
        hosted = self_hosted_prefixes((meta.get("spec") or {}).get("providers") or {})
        if hosted:
            out["self_hosted"] = hosted
        if meta.get("parent_run"):
            out["parent_run"] = meta["parent_run"]
            out["fork_round"] = meta.get("fork_round")
        return out

    def __repr__(self) -> str:
        return f"Run({self.id!r}, status={self.status!r}, dir={str(self.dir)!r})"

    # ---- M4: export (docs/INTERFACE-M4.md §1) -------------------------------------------------
    def export(self, out: Path | str | None = None) -> Path:
        """Tables, pi sessions and raw copies into `out` (default `<run dir>/export`)."""
        from .export import export_run

        return export_run(self.dir, out)


def self_hosted_prefixes(providers: dict) -> list[str]:
    """Model prefixes whose provider spec is self-hosted (`params.self_hosted`, or an
    OpenAI-compatible provider named `vllm` that does not say otherwise)."""
    out = []
    for prefix, spec in providers.items():
        params = (spec or {}).get("params") or {}
        flag = params.get("self_hosted")
        if flag is True or (flag is None and params.get("name") == "vllm"):
            out.append(prefix)
    return sorted(out)


def participant_model(p: Any) -> str | None:
    """The `"<prefix>:<id>"` model a participant calls, or None (scripted participants)."""
    model = getattr(p, "model", None)
    if not isinstance(model, str):
        model = (getattr(p, "params", None) or {}).get("model")
    return model if isinstance(model, str) and ":" in model else None
