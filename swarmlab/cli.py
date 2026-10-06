"""The `swarmlab` command line (docs/INTERFACE.md §16).

Every command builds an `Experiment` from YAML or opens a `Run` directory and calls the same
public API as Python (`Experiment.run`, `Run.load`, `Run.resume`, `Run.fork(...).run`,
`Run.view`); nothing here touches the Runner.

Decisions where the contract is silent:

- `--json` prints exactly one JSON object on stdout. For run-producing commands it is the run
  summary: `run_dir, run_id, spec_hash, status, end_reason, last_round, score, metrics` (metrics
  are the last logged value per metric name, `{"value", "denominator", "round"}`), plus
  command-specific keys (`replay: "ok"`, `view`, `parent_run`/`fork_round`). On failure it prints
  `{"ok": false, "error", "exit_code"}` and the error also goes to stderr.
- Exit codes: 0 success; 2 for a spec/validation problem (`SpecError`, an unknown arm, a plugin
  that cannot be resolved or constructed, a missing `max_rounds`); 1 for anything else
  (including `ReplayMismatch`).
- `--arm` may be omitted when the YAML has exactly one arm. For `fork --spec`, the arm defaults
  to the parent's arm when the edited YAML has it, else to the only arm.
- `validate` resolves every arm through `Experiment.from_yaml`, so unknown plugin types and bad
  constructor params are caught, not just YAML shape.
- M1b (WP7): `run` prints `Experiment.estimate(seed, max_rounds, calls_per_turn=...)` before
  running whenever any budget field is non-zero: one line `estimate: arm=... worst-case $X ...`
  on stdout, or on stderr under `--json` (stdout keeps exactly one JSON object). The estimate uses
  `calls_per_turn = options.max_calls_per_turn` (every turn hitting the runner's cap: the worst
  case), plus one call per probed agent per probed round. `swarmlab estimate SPEC [--arm A]
  [--seed N] [--max-rounds R] [--calls-per-turn C]` prints the same dict (default
  calls_per_turn: the worst case as above). `resume` takes `--budget-soft/--budget-hard/
  --budget-measurement`; given any of them, the effective budget (`run.json["budget"]`) is
  updated with those fields and passed to `Run.resume(budget=...)`.
"""
from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer

from .experiment import Experiment, Run
from .spec import Budget, RunOptions, SpecError, load_experiment_yaml

app = typer.Typer(
    name="swarmlab",
    help="Run, replay, resume, fork, and view swarmlab experiments.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)

JsonOpt = Annotated[bool, typer.Option("--json", help="Print one JSON object on stdout.")]


# ---- helpers ---------------------------------------------------------------------------------
def _build(spec: Path, arm: str | None) -> Experiment:
    """`Experiment.from_yaml`, with plugin resolution failures reported as `SpecError`."""
    doc = load_experiment_yaml(spec)
    arms = sorted(doc["arms"])
    if arm is None:
        if len(arms) != 1:
            raise SpecError(f"{spec}: --arm is required; available: {arms}")
        arm = arms[0]
    if arm not in doc["arms"]:
        raise SpecError(f"{spec}: unknown arm {arm!r}; available: {arms}")
    try:
        return Experiment.from_yaml(spec, arm)
    except SpecError:
        raise
    except (ValueError, TypeError, ImportError, AttributeError) as e:
        raise SpecError(f"{spec}: arm {arm!r}: {type(e).__name__}: {e}") from e


def summary(run: Run) -> dict[str, Any]:
    meta = run.meta
    metrics = {
        name: {"value": vals[-1][1], "denominator": vals[-1][2], "round": vals[-1][0]}
        for name, vals in run.metrics.items()
        if vals
    }
    out = {
        "run_dir": str(run.dir),
        "run_id": run.id,
        "spec_hash": meta.get("spec_hash"),
        "status": run.status,
        "end_reason": run.end_reason,
        "last_round": meta.get("last_round"),
        "score": run.score,
        "metrics": metrics,
    }
    if meta.get("parent_run"):
        out["parent_run"] = meta["parent_run"]
        out["fork_round"] = meta.get("fork_round")
    return out


def _fmt(v: Any) -> str:
    return f"{v:.3g}" if isinstance(v, float) else str(v)


def _human(data: dict[str, Any]) -> str:
    if "run_id" not in data:
        return "\n".join(f"{k}: {v}" for k, v in data.items())
    lines = [
        (f"run {data['run_id']}  status={data['status']}  end={data['end_reason']}  "
         f"rounds={data['last_round']}"),
        f"  dir    {data['run_dir']}",
        f"  spec   {str(data['spec_hash'])[:16]}",
        "  score  " + ", ".join(f"{k}={_fmt(v)}" for k, v in (data["score"] or {}).items()),
    ]
    if data["metrics"]:
        lines.append("  metrics " + ", ".join(
            f"{k}={_fmt(m['value'])}" for k, m in data["metrics"].items()))
    for key in ("parent_run", "fork_round", "replay", "view"):
        if key in data:
            lines.append(f"  {key} {data[key]}")
    return "\n".join(lines)


def _execute(fn: Callable[[], dict[str, Any]], as_json: bool) -> None:
    try:
        data = fn()
    except SpecError as e:
        _fail(e, 2, as_json)
    except (typer.Exit, typer.Abort):
        raise
    except Exception as e:  # noqa: BLE001 - the CLI reports every failure the same way
        _fail(e, 1, as_json)
    typer.echo(json.dumps(data, default=str) if as_json else _human(data))


def _fail(e: BaseException, code: int, as_json: bool) -> None:
    msg = f"{type(e).__name__}: {e}"
    print(f"error: {msg}", file=sys.stderr)
    if as_json:
        typer.echo(json.dumps({"ok": False, "error": msg, "exit_code": code}))
    raise typer.Exit(code)


def _estimate(exp: Experiment, seed: int, max_rounds: int | None,
              calls_per_turn: int | None = None) -> dict[str, Any]:
    rounds = max_rounds if max_rounds is not None else exp.options.get("max_rounds")
    if rounds is None:
        raise SpecError("no max_rounds in options; pass --max-rounds")
    if calls_per_turn is None:
        calls_per_turn = int(exp.options.get("max_calls_per_turn",
                                             RunOptions.model_fields["max_calls_per_turn"].default))
    return {**exp.estimate(seed, int(rounds), calls_per_turn=calls_per_turn),
            "calls_per_turn": calls_per_turn}


def _estimate_line(est: dict[str, Any]) -> str:
    by_model = ", ".join(f"{m}=${v:.4f}" for m, v in est["by_model"].items()) or "no model calls"
    return (f"estimate: arm={est['arm']} worst-case ${est['usd']:.4f} "
            f"({est['llm_agents']} model agents x {est['rounds']} rounds x "
            f"{est['calls_per_turn']} calls/turn, {est['probe_calls']} probe calls; {by_model}); budget {est['budget']}")


# ---- commands --------------------------------------------------------------------------------
@app.command()
def validate(
    spec: Annotated[Path, typer.Argument(help="Experiment YAML.")],
    as_json: JsonOpt = False,
) -> None:
    """Validate an experiment YAML and resolve every plugin of every arm."""

    def go() -> dict[str, Any]:
        doc = load_experiment_yaml(spec)
        arms = {}
        for arm in doc["arms"]:
            exp = _build(spec, arm)
            arms[arm] = {
                "world": exp.world.spec()["type"],
                "n_agents": len(exp.participants),
                "topology": exp.medium.spec()["params"]["topology"],
                "metrics": [m if isinstance(m, str) else m.name for m in exp.metrics],
                "max_rounds": exp.options.get("max_rounds"),
            }
        return {"ok": True, "spec": str(spec), "name": doc["name"], "arms": arms}

    _execute(go, as_json)


@app.command()
def run(
    spec: Annotated[Path, typer.Argument(help="Experiment YAML.")],
    seed: Annotated[int, typer.Option("--seed", help="Run seed.")],
    arm: Annotated[str | None, typer.Option("--arm", help="Arm name (optional if only one).")] = None,
    max_rounds: Annotated[int | None, typer.Option("--max-rounds", help="Override options.max_rounds.")] = None,
    out: Annotated[Path, typer.Option("--out", help="Parent directory for run dirs.")] = Path("runs"),
    as_json: JsonOpt = False,
) -> None:
    """Run one arm of an experiment with one seed."""

    def go() -> dict[str, Any]:
        exp = _build(spec, arm)
        if max_rounds is None and "max_rounds" not in exp.options:
            raise SpecError(f"{spec}: no max_rounds in options; pass --max-rounds")
        if any(v > 0 for v in exp.budget.model_dump().values()):
            line = _estimate_line(_estimate(exp, seed, max_rounds))
            if as_json:
                print(line, file=sys.stderr)
            else:
                typer.echo(line)
        return summary(exp.run(seed=seed, max_rounds=max_rounds, out=out))

    _execute(go, as_json)


@app.command()
def estimate(
    spec: Annotated[Path, typer.Argument(help="Experiment YAML.")],
    arm: Annotated[str | None, typer.Option("--arm", help="Arm name (optional if only one).")] = None,
    seed: Annotated[int, typer.Option("--seed", help="Run seed.")] = 0,
    max_rounds: Annotated[int | None, typer.Option("--max-rounds", help="Override options.max_rounds.")] = None,
    calls_per_turn: Annotated[int | None, typer.Option(
        "--calls-per-turn", help="Model calls per turn (default: options.max_calls_per_turn).")] = None,
    as_json: JsonOpt = False,
) -> None:
    """Print a rough worst-case dollar estimate for one arm (no run, no provider call)."""

    def go() -> dict[str, Any]:
        est = _estimate(_build(spec, arm), seed, max_rounds, calls_per_turn)
        return est if as_json else {"estimate": _estimate_line(est)[len("estimate: "):]}

    _execute(go, as_json)


@app.command()
def replay(
    run_dir: Annotated[Path, typer.Argument(help="Run directory.")],
    as_json: JsonOpt = False,
) -> None:
    """Replay a run from its log and check metrics and score (exit 1 on a mismatch)."""
    _execute(lambda: {**summary(Run.load(run_dir)), "replay": "ok"}, as_json)


@app.command()
def resume(
    run_dir: Annotated[Path, typer.Argument(help="Run directory.")],
    budget_soft: Annotated[float | None, typer.Option("--budget-soft", help="New soft budget (USD).")] = None,
    budget_hard: Annotated[float | None, typer.Option("--budget-hard", help="New hard ceiling (USD).")] = None,
    budget_measurement: Annotated[float | None, typer.Option(
        "--budget-measurement", help="New measurement budget (USD).")] = None,
    as_json: JsonOpt = False,
) -> None:
    """Resume an interrupted (or budget-ended) run from its last committed round."""

    def go() -> dict[str, Any]:
        r = Run(run_dir)
        changes = {k: v for k, v in (("soft_usd", budget_soft), ("hard_usd", budget_hard),
                                     ("measurement_usd", budget_measurement)) if v is not None}
        budget = None
        if changes:
            current = r.meta.get("budget") or r.meta["spec"]["budget"]
            budget = Budget(**{**current, **changes})
        return summary(r.resume(budget=budget))

    _execute(go, as_json)


@app.command()
def fork(
    run_dir: Annotated[Path, typer.Argument(help="Parent run directory.")],
    at: Annotated[int, typer.Option("--at", help="Committed round to fork from.")],
    spec: Annotated[Path | None, typer.Option("--spec", help="Edited experiment YAML.")] = None,
    arm: Annotated[str | None, typer.Option("--arm", help="Arm of the edited YAML.")] = None,
    max_rounds: Annotated[int | None, typer.Option("--max-rounds", help="Override max_rounds.")] = None,
    out: Annotated[Path | None, typer.Option("--out", help="Parent directory for the fork.")] = None,
    as_json: JsonOpt = False,
) -> None:
    """Fork a run at a committed round and continue it live (optionally under an edited spec)."""

    def go() -> dict[str, Any]:
        parent = Run(run_dir)
        experiment = None
        if spec is not None:
            chosen = arm
            if chosen is None:
                arms = load_experiment_yaml(spec)["arms"]
                if parent.meta.get("arm") in arms:
                    chosen = parent.meta["arm"]
            experiment = _build(spec, chosen)
        child = parent.fork(at, experiment=experiment).run(out=out, max_rounds=max_rounds)
        return summary(child)

    _execute(go, as_json)


@app.command()
def view(
    run_dir: Annotated[Path, typer.Argument(help="Run directory.")],
    as_json: JsonOpt = False,
) -> None:
    """Build the self-contained replay page `view.html` for a run."""

    def go() -> dict[str, Any]:
        r = Run(run_dir)
        return {**summary(r), "view": str(r.view())}

    _execute(go, as_json)


def main() -> None:  # pragma: no cover - `python -m swarmlab.cli`
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
