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
- When a `Run` came from `Experiment.run`, `fork`/`resume` reuse that in-memory experiment
  (so plugins defined inline in a script work); a loaded `Run` rebuilds it with `from_spec`.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .events import Event, EventLog, logical_view
from .ids import run_id as make_run_id
from .medium.board import Board
from .metrics.base import Metric
from .participants.base import Participant
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

    # ---- operations --------------------------------------------------------------------------
    def view(self) -> Path:
        from .viewer.build import build as build_view

        return build_view(self.dir)

    def fork(self, at_round: int, experiment: Experiment | None = None) -> ForkHandle:
        return ForkHandle(self, at_round, experiment)

    def resume(self) -> Run:
        Runner(self.dir, self.experiment, self.spec.options).resume()
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
