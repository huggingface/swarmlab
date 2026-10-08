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
- `from_yaml` puts the YAML's directory at the front of `sys.path` before resolving plugins
  (`registry.add_import_dir`), so `module:Class` plugins in files next to the spec work without
  `PYTHONPATH`; the directory is kept on the experiment and recorded by the runner in
  `run.json["import_dir"]`, which `Run.experiment` puts on `sys.path` before `from_spec`.
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
- M6: `estimate` scales each participant's turns by the scheduler's `turns_per_round(n) / n`
  (`options.scheduler`; 1 for SeededShuffle, 1/n for OneSpeaker) and caps calls per turn by a
  participant's `calls_per_turn_cap` (LLMAgent: `max_calls`, or 2 under `report_json`) when it
  has one, else its `max_calls`.
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
  `total_usd_flat` accompanies `total_usd` when `seeds` is given. `usd_per_round` is
  `usd_flat / max_rounds` (the CLI compares it with the gap between `soft_usd` and `hard_usd`).
  `calls_per_turn_used` lists the distinct calls per turn actually priced (`calls_per_turn`
  capped by each model-backed participant's `max_calls`). `usd_per_round_max` is the last
  round's cost (growth included).
- Measured estimates (field notes item 5): `Run.measured()` reads a finished run's figures (mean
  model calls per model-backed turn, prompt tokens per call fitted as a line over rounds,
  completion tokens per call, measurement spend per answered probe call); `estimate` takes
  them as `calls_per_turn`, `prompt_tokens` (the line's round-1 value), `prompt_growth` (its
  slope, >= 0), `completion_tokens` (floats are fine) and `probe_call_usd` (a probe call's
  price instead of the token formula).
- `run_all` honours `budget.total_usd` (experiment-wide): before each run it would start,
  `admit(seed, out)` checks the experiment ledger `<out>/<name>.ledger.jsonl` (spend of every
  run of the experiment under `out`, from any process or invocation, plus the headroom of runs
  still in flight, plus this arm's `hard_usd` must fit; `budget.ExperimentLedger`). A run refused
  only for headroom held by runs in flight waits for them (`admit_or_wait`, polling every
  `budget.ADMIT_POLL_S`); a final refusal stops and warns (`TotalBudgetWarning`, naming the
  skipped seeds). Run dirs it finds are backfilled
  into the ledger when they have no row (written before the ledger existed).
- `Run.summary()` is the CLI's run summary: run_dir, run_id, arm, seed, spec_hash, status,
  end_reason, last_round, score, metrics (last value per name), spend, `probes_skipped`
  (`{reason: count}`, only when the spec has probes; `Run.probes_skipped()`) (+ parent_run/fork_round
  for a fork; + `turns_total`, `turns_errored`, and `first_error` and `health: "degraded"` when
present: more than half of the turns ended `error`, see swarmlab/runner.py; + `self_hosted`: the model prefixes served self-hosted, e.g. `["vllm"]`, only when
  there are any, so a $0 spend reads as compute time, not free). `Run.load(dir, run_id=None)` accepts a run dir, or a parent dir plus run id.
- M3a §2: `run(..., repeat=None, run_id=None)`: `repeat` is `RunOptions.repeat` (paired-run
  sampling repeat, see swarmlab/runner.py); `run_id` overrides the derived run id (and, without
  `run_dir`, the directory name). `pair(seed, max_rounds, *, patch, repeats=1, control=True,
  out="runs", **run_options) -> PairedResult`: see `pair` and `PairedResult`. A `Run.fork(0, exp)`
  re-resets the world under `exp` with the parent's seed (swarmlab/runner.py).
"""
from __future__ import annotations

import json
import time
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from . import budget as budget_mod
from ._io import atomic_write_bytes
from .budget import Admission, ExperimentLedger, TotalBudgetWarning
from .events import Event, EventLog, logical_view
from .ids import run_id as make_run_id
from .interventions import build_intervention, check_names
from .medium.board import Board
from .metrics.base import Metric
from .participants.base import Participant
from .probes import Probe, build_probe
from .providers import catalog_priced, ensure_pricing
from .providers import resolve as resolve_provider
from .providers.base import Provider, split_model
from .registry import add_import_dir, build
from .roles import Role, assign, role_name, roles_from_spec, spec_roles
from .runner import Runner
from .scheduler import build_scheduler
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
    resolve_rounds,
    runspec_to_doc,
    spec_hash,
    unbilled_spec,
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
    roles: dict[str, Role] = {}  # M3c: swarmlab/roles.py (assign agents with roles.assign)
    _resolved: dict[str, Provider] = PrivateAttr(default_factory=dict)
    _import_dir: str | None = PrivateAttr(default=None)  # the YAML's dir (registry.add_import_dir)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.probes = [p if isinstance(p, Probe) else build_probe(p) for p in self.probes]
        names = [p.name for p in self.probes]
        if len(set(names)) != len(names):
            raise ValueError(f"probe names must be unique, got {names}")
        self.interventions = [build_intervention(i) for i in self.interventions]  # M3a
        check_names(self.interventions)
        # after validation, so UnknownModelPricing propagates as itself (not a ValidationError)
        models = [participant_model(p) for p in self.participants]
        models += [p.coder_model() for p in self.probes]
        models += [r.model for r in self.roles.values()]  # M3c: role model overrides
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

    def estimate(self, seed: int, max_rounds: int, calls_per_turn: float = 2,
                 prompt_tokens: float = 3000, completion_tokens: float = 300,
                 seeds: list[int] | None = None, prompt_growth: float = 0,
                 probe_call_usd: float | None = None) -> dict:
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
        cpts: set[float] = set()
        last_round_extra = 0.0  # growth adds this much to the last round over a flat round
        # M6: a scheduler may run fewer turns per round than agents (OneSpeaker: one); each
        # agent then takes `share` of the rounds' turns in expectation
        n_agents = len(self.participants)
        sched = build_scheduler(self.options.get("scheduler"))
        share = sched.turns_per_round(n_agents) / n_agents if n_agents else 0.0
        for p in self.participants:
            model = participant_model(p)
            if model is None:
                continue
            llm += 1
            provider, mid = self.provider_for(model)
            p_in, p_out, _ = provider.model_pricing(mid)
            own = getattr(p, "calls_per_turn_cap", getattr(p, "max_calls", None))
            cpt = min(calls_per_turn, own) if isinstance(own, int) and own > 0 else calls_per_turn
            cpts.add(cpt)
            calls += cpt * max_rounds * share
            last_round_extra += cpt * share * prompt_growth * (max_rounds - 1) * p_in / 1e6
            out_usd = completion_tokens * p_out * max_rounds
            by_model[model] = (by_model.get(model, 0.0)
                               + share * cpt * (grown_tokens * p_in + out_usd) / 1e6)
            by_model_flat[model] = (by_model_flat.get(model, 0.0)
                                    + share * cpt * (flat_tokens * p_in + out_usd) / 1e6)
            if callable(getattr(p, "probe_context", None)):
                for probe in self.probes:
                    every = max(1, probe.every)
                    probed = [r for r in rounds if r % every == 0]
                    probe_calls += len(probed)
                    if probe_call_usd is not None:  # measured in a finished run
                        measurement += probe_call_usd * len(probed)
                        measurement_flat += probe_call_usd * len(probed)
                        continue
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
        out["usd_per_round"] = out["usd_flat"] / max(1, max_rounds)  # one round, no growth
        out["usd_per_round_max"] = out["usd_per_round"] + last_round_extra  # the last round
        out["calls_per_turn_used"] = sorted(cpts)  # after each participant's own max_calls
        if seeds is not None:
            out["runs"] = len(seeds)
            out["total_usd"] = out["usd"] * len(seeds)
            out["total_usd_flat"] = out["usd_flat"] * len(seeds)
        return out

    # ---- running -----------------------------------------------------------------------------
    def _options(self, seed: int, max_rounds: int | None, **kw: Any) -> RunOptions:
        merged = {k: v for k, v in self.options.items() if k != "seed"}
        merged.update({k: v for k, v in kw.items() if v is not None})
        merged = resolve_rounds(merged, len(self.participants))  # M6: rounds_per_agent
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
        repeat: int | None = None,
        run_id: str | None = None,
    ) -> Run:
        """Run one seed into `out/<run_id>` (or exactly `run_dir` when given)."""
        options = self._options(seed, max_rounds, commit=commit, max_calls_per_turn=max_calls_per_turn,
                                snapshot_every=snapshot_every, concurrency=concurrency, repeat=repeat)
        rid = run_id or self.run_id(seed)
        target = Path(run_dir) if run_dir is not None else Path(out) / rid
        Runner(target, self, options, run_id=run_id).live()
        return Run(target, experiment=self)

    def pair(self, seed: int, max_rounds: int | None = None, *, patch: Experiment | None,
             repeats: int = 1, control: bool = True, out: Path | str = "runs",
             **run_options: Any) -> PairedResult:
        """Paired runs from round 0 (M3a §2): per repeat i, the base run, the `patch` run and
        (with `control`) an unchanged re-run of the base, all with the same seed and repeat i.

        Layout: `<out>/<run_id(seed)>__pair/` holds `pair.json` and one run dir per run, named
        (and with run id) `<run_id(seed)>__{base,patched,control}_r<i>`. Repeat 0 of the base is
        exactly `run(seed)` (only the run id differs). `patch=None` pairs the base with itself.
        """
        if repeats < 1:
            raise ValueError("repeats must be >= 1")
        patched_exp = self if patch is None else patch
        if len(patched_exp.participants) != len(self.participants):
            raise ValueError("a paired run cannot change the number of participants")
        base_id = self.run_id(seed)
        root = Path(out) / f"{base_id}__pair"
        if (root / "pair.json").exists():
            raise FileExistsError(f"{root} already holds a paired run; use another out dir")
        root.mkdir(parents=True, exist_ok=True)
        roles: list[tuple[str, Experiment]] = [("base", self), ("patched", patched_exp)]
        if control:
            roles.append(("control", self))
        runs: dict[str, list[Run]] = {"base": [], "patched": [], "control": []}
        for i in range(repeats):
            for role, exp in roles:
                rid = f"{base_id}__{role}_r{i}"
                runs[role].append(exp.run(seed, max_rounds, run_dir=root / rid, run_id=rid,
                                          repeat=i, **run_options))
        result = PairedResult(root, runs["base"], runs["patched"], runs["control"])
        doc = {"seed": seed, "repeats": repeats, "control": control,
               "base_spec_hash": runs["base"][0].meta["spec_hash"],
               "patched_spec_hash": runs["patched"][0].meta["spec_hash"],
               "runs": {k: [r.dir.name for r in v] for k, v in result.runs.items()}}
        atomic_write_bytes(root / "pair.json", (json.dumps(doc, indent=2) + "\n").encode())
        return result

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

    def experiment_ledger(self, out: Path | str = "runs") -> ExperimentLedger:
        """`<out>/<name>.ledger.jsonl`: spend of every run of this experiment under `out`."""
        return ExperimentLedger(out, self.name)

    def unbilled(self) -> bool:
        """Nothing this experiment calls is billed (only `fake:` models; `spec.unbilled_spec`)."""
        return unbilled_spec(self.to_spec(0, 1).model_dump(mode="json"))

    def admit(self, seed: int, out: Path | str = "runs") -> Admission:
        """The `budget.total_usd` start check for the run of `seed` under `out` (see
        `ExperimentLedger.admit`): refusal (None when admitted), spend and reserved headroom it
        was checked against, the runs counted in flight, and whether a refusal only has to wait
        for them. An unbilled experiment (only `fake:` models) is always admitted."""
        total = self.budget.total_usd
        if total <= 0 or self.unbilled():
            return Admission(None, 0.0, 0.0)
        led = self.experiment_ledger(out)
        led.backfill_dir(out)  # run dirs from before the ledger existed
        return led.admit(self.run_id(seed), self.budget.hard_usd, total)

    def admit_or_wait(self, seed: int, out: Path | str = "runs",
                      on_wait: Callable[[Admission], None] | None = None,
                      sleep: Callable[[float], Any] = time.sleep) -> Admission:
        """`admit`, retried every `budget.ADMIT_POLL_S` while the refusal only waits for runs
        in flight (`Admission.wait`); `on_wait` is called with the first waiting answer and
        whenever the runs in flight change. Returns the admission or the final refusal."""
        seen: tuple[str, ...] | None = None
        while True:
            adm = self.admit(seed, out)
            if not adm.wait:
                return adm
            ids = tuple(f.run_id for f in adm.in_flight)
            if on_wait is not None and ids != seen:
                on_wait(adm)
            seen = ids
            sleep(budget_mod.ADMIT_POLL_S)

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
        led = self.experiment_ledger(out)
        for i, seed in enumerate(seeds):
            state, found = self.existing(seed, max_rounds, out, **run_options)
            if found is not None:
                led.backfill(found.meta)
            if state == "same" and skip_existing and not rerun and found is not None:
                runs.append(found)
                continue
            if state == "different" and not rerun and found is not None:
                raise FileExistsError(
                    f"{found.dir} holds a run with a different spec_hash; pass rerun=True "
                    "(CLI: --rerun) or another out dir")
            why = self.admit_or_wait(seed, out).refusal
            if why:  # final: the later seeds have the same hard_usd
                warnings.warn(TotalBudgetWarning(f"total cap: {why}; skipped seeds {seeds[i:]}"),
                              stacklevel=2)
                break
            target = self.rerun_dir(out, self.run_id(seed)) if rerun and state != "new" else None
            try:
                runs.append(self.run(seed, max_rounds, out=out, run_dir=target, **run_options))
            except BaseException:
                led.record(self.run_id(seed), f"{self.run_id(seed)}@admit", 0.0, "failed")
                raise
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
        roles, participant_roles = spec_roles(self.roles, [role_name(p) for p in self.participants])
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
            interventions=[PluginSpec(**i.spec()) for i in self.interventions],
            roles=roles,
            participant_roles=participant_roles,
        )

    @classmethod
    def from_spec(cls, spec: RunSpec) -> Experiment:
        medium = spec.medium.model_dump(mode="json")
        return cls(
            name=spec.experiment,
            arm=spec.arm,
            world=build(spec.world, "swarmlab.worlds"),
            participants=[assign(build(p, "swarmlab.participants"), r) for p, r in
                          zip(spec.participants, [*spec.participant_roles, *[None] * len(spec.participants)],
                              strict=False)],
            roles=roles_from_spec(spec.roles),
            medium=Board(**medium),
            metrics=[build(m, "swarmlab.metrics") for m in spec.metrics],
            budget=spec.budget,
            providers={k: build(p, "swarmlab.providers") for k, p in spec.providers.items()} or None,
            probes=[build(p, "swarmlab.probes") for p in spec.probes],
            interventions=list(spec.interventions),
        )

    @classmethod
    def from_yaml(cls, path: Path | str, arm: str) -> Experiment:
        doc = load_experiment_yaml(path)
        if arm not in doc["arms"]:
            raise SpecError(f"unknown arm {arm!r}; available: {sorted(doc['arms'])}")
        options = {**doc["options"], **doc["arms"][arm]["options"]}
        options.pop("seed", None)
        n_agents = sum(g["count"] for g in doc["arms"][arm]["participants"])
        options = resolve_rounds(options, n_agents)  # M6: rounds_per_agent -> max_rounds
        spec = arm_to_runspec(doc, arm, seed=0, max_rounds=options.get("max_rounds", 1))
        import_dir = add_import_dir(Path(path).resolve().parent)  # module:Class next to the YAML
        exp = cls.from_spec(spec)
        exp.options = options
        exp._import_dir = import_dir
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


class PairedResult:
    """The runs of `Experiment.pair` and their paired metric differences.

    `runs` is `{"base": [...], "patched": [...], "controls": [...]}` (one `Run` per repeat;
    `controls` is empty without `control`). `effect(metric, round=-1)` compares the metric at
    `round` (-1: each run's last logged value) per repeat: `diff` is the mean of patched - base,
    `control_spread` is the root-mean-square of the control diffs (control - base), i.e. their
    spread around the zero they should have; it is defined for a single control pair (= its
    absolute diff) and is None without controls. Repeats whose value is None on either side are
    left out (`n` / `n_control` count the pairs used).
    """

    def __init__(self, dir: Path, base: list[Run], patched: list[Run], controls: list[Run]) -> None:
        self.dir = dir
        self.base = base
        self.patched = patched
        self.controls = controls

    @property
    def runs(self) -> dict[str, list[Run]]:
        return {"base": self.base, "patched": self.patched, "controls": self.controls}

    @staticmethod
    def value(run: Run, metric: str, round: int = -1) -> float | None:
        series = run.metrics.get(metric)
        if not series:
            raise KeyError(f"{run.id} logged no metric {metric!r}")
        if round == -1:
            return series[-1][1]
        for r, v, _ in series:
            if r == round:
                return v
        raise KeyError(f"{run.id} has no {metric!r} value at round {round}")

    def _diffs(self, others: list[Run], metric: str, round: int) -> list[float]:
        out = []
        for b, o in zip(self.base, others, strict=False):
            vb, vo = self.value(b, metric, round), self.value(o, metric, round)
            if vb is not None and vo is not None:
                out.append(vo - vb)
        return out

    def effect(self, metric: str, round: int = -1) -> dict[str, Any]:
        diffs = self._diffs(self.patched, metric, round)
        cdiffs = self._diffs(self.controls, metric, round)
        return {
            "metric": metric, "round": round,
            "diff": sum(diffs) / len(diffs) if diffs else None,
            "control_spread": (sum(d * d for d in cdiffs) / len(cdiffs)) ** 0.5 if cdiffs else None,
            "diffs": diffs, "control_diffs": cdiffs, "n": len(diffs), "n_control": len(cdiffs),
        }

    def __repr__(self) -> str:
        return f"PairedResult({str(self.dir)!r}, repeats={len(self.base)}, controls={len(self.controls)})"


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
            if self._meta.get("import_dir"):  # the spec's dir: module:Class plugins next to it
                add_import_dir(self._meta["import_dir"])
            self._experiment = Experiment.from_spec(self.spec)
            self._experiment._import_dir = self._meta.get("import_dir")
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

    def probes_skipped(self) -> dict[str, int]:
        """Skipped probe events by reason (`parsed["skipped"]`: `no_context`,
        `measurement_budget`, `hard_ceiling`, `provider_error`) in the log."""
        out: dict[str, int] = {}
        for ev in self.events_all:
            if ev.type == "probe" and "skipped" in (ev.parsed or {}):
                why = str(ev.parsed["skipped"])
                out[why] = out.get(why, 0) + 1
        return dict(sorted(out.items()))

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
        out.update(self.turn_health())
        if (meta.get("spec") or {}).get("probes"):
            out["probes_skipped"] = self.probes_skipped()
        hosted = self_hosted_prefixes((meta.get("spec") or {}).get("providers") or {})
        if hosted:
            out["self_hosted"] = hosted
        if meta.get("parent_run"):
            out["parent_run"] = meta["parent_run"]
            out["fork_round"] = meta.get("fork_round")
        return out

    def measured(self) -> dict[str, Any]:
        """Per-call figures measured in this run, for `Experiment.estimate` (module doc):
        `calls_per_turn`, `prompt_tokens`, `prompt_growth`, `completion_tokens`,
        `probe_call_usd` (None without answered probes), plus `run_id`, `rounds`, `turns`,
        `calls`, `per_round` (`[(round, mean prompt tokens per call)]`). Raises `ValueError`
        when the run made no swarm model call."""
        from .ids import agent_id
        from .spec import spec_models

        spec = self.meta.get("spec") or {}
        model_agents = {str(agent_id(i)) for i, p in enumerate(spec.get("participants") or [])
                        if spec_models({"participants": [p]})}
        turns = calls = 0
        prompt = completion = 0
        by_round: dict[int, list[int]] = {}
        probes = 0
        for ev in self.events_all:
            if ev.type == "turn_ended" and str(ev.agent) in model_agents:
                usage = ev.usage or {}
                n = int(usage.get("inference_calls") or 0)
                turns += 1
                calls += n
                prompt += int(usage.get("prompt_tokens") or 0)
                completion += int(usage.get("completion_tokens") or 0)
                if n:
                    row = by_round.setdefault(ev.round, [0, 0])
                    row[0] += int(usage.get("prompt_tokens") or 0)
                    row[1] += n
            elif ev.type == "probe" and "skipped" not in (ev.parsed or {}):
                probes += 1
        if not calls:
            raise ValueError(f"{self.id} made no swarm model call; nothing to measure")
        points = sorted((r, p / n) for r, (p, n) in by_round.items())
        intercept, slope = _fit_line([(r - 1, y) for r, y in points])
        if slope < 0:  # a shrinking context (windowed memory): price it flat at the mean
            intercept, slope = sum(y for _, y in points) / len(points), 0.0
        measurement = float(self.spend.get("measurement") or 0)
        return {"run_id": self.id, "rounds": int(self.meta.get("last_round") or 0),
                "turns": turns, "calls": calls, "calls_per_turn": calls / max(1, turns),
                "prompt_tokens": max(0.0, intercept), "prompt_growth": slope,
                "completion_tokens": completion / calls,
                "probe_call_usd": measurement / probes if probes else None,
                "per_round": [(r, round(y, 1)) for r, y in points]}

    def turn_health(self) -> dict[str, Any]:
        """`turns_total`, `turns_errored`, and `first_error` / `health: "degraded"` when present
        (from run.json; counted from the log for run dirs written before these fields existed)."""
        meta = self.meta
        if "turns_total" in meta:
            data = {k: meta[k] for k in ("turns_total", "turns_errored", "first_error", "health")
                    if k in meta}
        else:
            from .runner import TurnTally

            data = TurnTally.of(list(self.events_all)).meta()
        return data

    def __repr__(self) -> str:
        return f"Run({self.id!r}, status={self.status!r}, dir={str(self.dir)!r})"

    # ---- M4: export (docs/INTERFACE-M4.md §1) -------------------------------------------------
    def export(self, out: Path | str | None = None) -> Path:
        """Tables, pi sessions and raw copies into `out` (default `<run dir>/export`)."""
        from .export import export_run

        return export_run(self.dir, out)


def _fit_line(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Least-squares (intercept, slope) through `(x, y)` points; slope 0 for a single x."""
    n = len(points)
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    sxx = sum((x - mx) ** 2 for x, _ in points)
    if sxx == 0:
        return my, 0.0
    slope = sum((x - mx) * (y - my) for x, y in points) / sxx
    return my - slope * mx, slope


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
