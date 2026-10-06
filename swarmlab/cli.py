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
"""
from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer

from .experiment import Experiment, Run
from .spec import SpecError, load_experiment_yaml

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
        return summary(exp.run(seed=seed, max_rounds=max_rounds, out=out))

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
    as_json: JsonOpt = False,
) -> None:
    """Resume an interrupted run from its last committed round."""
    _execute(lambda: summary(Run(run_dir).resume()), as_json)


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
