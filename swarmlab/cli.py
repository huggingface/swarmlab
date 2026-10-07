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
  `calls_per_turn = --calls-per-turn` if given, else `options.max_calls_per_turn` (every turn
  hitting the runner's cap: the worst case), capped per participant by its own `max_calls`, plus
  one call per probed agent per probed round. The result's `calls_per_turn` is the value(s)
  actually priced (an int, or the distinct values when groups differ) and
  `calls_per_turn_source` says where it came from; the printed line and table show both. `swarmlab estimate SPEC [--arm A]
  [--seed N] [--max-rounds R] [--calls-per-turn C]` prints the same dict (default
  calls_per_turn: the worst case as above). `resume` takes `--budget-soft/--budget-hard/
  --budget-measurement`; given any of them, the effective budget (`run.json["budget"]`) is
  updated with those fields and passed to `Run.resume(budget=...)`. The values are run totals
  (the ledger, discarded rounds included, counts against them). `--add-budget X` sets
  soft = swarm spend + X and hard = swarm + measurement spend + X (only the enforced ones).
  `resume` prints the spend so far and the budget it resumes with (stderr under --json).
- Run summaries (status line of `run`, `resume`, `replay`, `fork`, ...) carry
  `spend=$x.xxx (swarm $a + measurement $b)` from the ledger; the JSON adds `spend_usd`, and
  `resume --json` adds the `budget` it resumed with.
- Setup UX: `run SPEC` without `--arm`/`--seed` runs every arm (document order) x every seed in
  the YAML's `seeds:` (default `[0]`); `--arm`/`--seed` narrow it. With one arm and a `--seed`
  (the M1a form) the output is the single run summary as before; otherwise it is
  `{"ok", "estimate": {"arms": {arm: est}, "total_usd", "runs"}, "runs": [summary + "outcome"]}`
  where outcome is `ran`, `skipped` (out/<run_id> holds a run with the same spec_hash; printed as
  "exists, skipping") or `failed` (with `error`). A run dir holding a different spec_hash is a
  failure unless `--rerun`, which writes `out/<run_id>__r<N>` (also for same-hash runs). Before
  running, the per-arm estimate and the total are printed (stderr under `--json`) whenever any
  arm has a non-zero budget or a model-backed participant. When any budget is non-zero the
  command asks for confirmation on stderr unless `--yes`; declining (or no TTY to answer)
  exits 1 before anything runs. A failed run does not stop the others; the exit code is then 1
  (2 if every failure was a SpecError).
- Total cap (`budget.total_usd`, top level): `run` prints the per-run caps of every arm and the
  total cap before anything runs, and warns per arm when `hard_usd - soft_usd` is less than one
  round of the estimate (`usd_per_round`). Before starting each run it checks
  `budget.total_cap_refusal(spend of the runs ran or found so far, the arm's hard_usd, total)`;
  on a refusal that run and every later one get outcome `capped`, the JSON gets
  `capped: {reason, total_usd, spent, skipped: [run ids]}` and the exit code is 1.
- `job run|status|logs|fetch` (WP8, HF Jobs with a co-located vLLM server) are documented in
  `swarmlab/jobs/` and docs/handoff/WP8.md. `job run` only prints unless `--launch`.
- `models`, `doctor`, `init` are documented in their own modules (`providers/catalog.py`,
  `doctor.py`) and in `init`'s help.
"""
from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer

from .budget import ledger_total, total_cap_refusal
from .experiment import Experiment, Run
from .spec import Budget, RunOptions, SpecError, experiment_seeds, load_experiment_yaml

app = typer.Typer(
    name="swarmlab",
    help="Set up, run, replay, resume, fork, and view swarmlab experiments.\n\n"
         "Quickstart: swarmlab doctor; swarmlab init demo; swarmlab run demo.yaml",
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
    """`Run.summary()` plus `spend_usd` (ledger swarm + measurement, discarded rounds included)."""
    data = run.summary()
    data["spend_usd"] = ledger_total(data.get("spend"))
    return data


def _spend_text(spend: dict | None) -> str:
    spend = spend or {}
    swarm, meas = float(spend.get("swarm") or 0), float(spend.get("measurement") or 0)
    return f"spend=${swarm + meas:.3f} (swarm ${swarm:.3f} + measurement ${meas:.3f})"


def _fmt(v: Any) -> str:
    return f"{v:.3g}" if isinstance(v, float) else str(v)


def _human(data: dict[str, Any]) -> str:
    if "text" in data:
        return str(data["text"])
    if "run_id" not in data:
        return "\n".join(f"{k}: {v}" for k, v in data.items())
    lines = [
        (f"run {data['run_id']}  status={data['status']}  end={data['end_reason']}  "
         f"rounds={data['last_round']}  {_spend_text(data.get('spend'))}"),
        f"  dir    {data['run_dir']}",
        f"  spec   {str(data['spec_hash'])[:16]}",
        "  score  " + ", ".join(f"{k}={_fmt(v)}" for k, v in (data["score"] or {}).items()),
    ]
    if data["metrics"]:
        lines.append("  metrics " + ", ".join(
            f"{k}={_fmt(m['value'])}" for k, m in data["metrics"].items()))
    for key in ("parent_run", "fork_round", "replay", "view", "skipped", "self_hosted", "fetched",
                "published"):
        if key in data:
            lines.append(f"  {key} {data[key]}")
    if isinstance(data.get("budget"), dict):
        lines.append("  " + _budget_text(data["budget"]))
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
              calls_per_turn: int | None = None, prompt_growth: int = 0) -> dict[str, Any]:
    rounds = max_rounds if max_rounds is not None else exp.options.get("max_rounds")
    if rounds is None:
        raise SpecError("no max_rounds in options; pass --max-rounds")
    cap = int(exp.options.get("max_calls_per_turn",
                              RunOptions.model_fields["max_calls_per_turn"].default))
    requested = calls_per_turn if calls_per_turn is not None else cap
    est = exp.estimate(seed, int(rounds), calls_per_turn=requested, prompt_growth=prompt_growth)
    used = est["calls_per_turn_used"] or [requested]
    source = (f"--calls-per-turn {calls_per_turn}" if calls_per_turn is not None
              else f"options.max_calls_per_turn {cap}")
    if used != [requested]:
        source = f"participant max_calls, under {source}"
    return {**est, "calls_per_turn": used[0] if len(used) == 1 else used,
            "calls_per_turn_source": source}


def _cpt_text(est: dict[str, Any]) -> str:
    cpt = est["calls_per_turn"]
    n = f"{min(cpt)}-{max(cpt)}" if isinstance(cpt, list) else str(cpt)
    return f"{n} calls/turn ({est.get('calls_per_turn_source', '')})"


def _budget_text(budget: dict[str, float]) -> str:
    parts = [f"{k.removesuffix('_usd')}=${v:g}" for k, v in budget.items() if v > 0]
    return "budget " + (" ".join(parts) if parts else "none (0 = not enforced)")


def _only_fake(ests: Any) -> bool:
    models = [m for est in ests for m in est["by_model"]]
    return bool(models) and all(m.startswith("fake:") for m in models)


def _estimate_line(est: dict[str, Any], seeds: int | None = None) -> str:
    by_model = ", ".join(f"{m}=${v:.4f}" for m, v in est["by_model"].items()) or "no model calls"
    per = f"worst-case ${est['usd']:.4f}"
    if seeds is not None:
        per += f" per run x {seeds} seed(s) = ${est['usd'] * seeds:.4f}"
    return (f"estimate: arm={est['arm']} {per} "
            f"({est['llm_agents']} model agents x {est['rounds']} rounds x "
            f"{_cpt_text(est)}, {est['probe_calls']} probe calls; {by_model}); "
            f"{_budget_text(est['budget'])}")


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


def _say(line: str, as_json: bool) -> None:
    """Progress output: stdout, or stderr under --json (stdout keeps one JSON object)."""
    print(line, file=sys.stderr if as_json else sys.stdout, flush=True)


def _has_budget(exp: Experiment) -> bool:
    return any(v > 0 for v in exp.budget.model_dump().values())


def _usd(v: Any) -> str:
    return f"${v:.4f}" if isinstance(v, (int, float)) else "-"


def _score_text(score: dict | None) -> str:
    return ", ".join(f"{k}={_fmt(v)}" for k, v in (score or {}).items()) or "-"


def _table(rows: list[dict[str, Any]]) -> str:
    head = ("run", "outcome", "end", "score", "spend")
    body = []
    for r in rows:
        spend = r.get("spend") or {}
        total = (spend.get("swarm") or 0) + (spend.get("measurement") or 0) if spend else None
        money = _usd(total)
        if r.get("self_hosted"):
            money += " +compute (self-hosted)"
        body.append((r.get("run_id") or "?", r["outcome"], str(r.get("end_reason") or "-"),
                     _score_text(r.get("score")) if r["outcome"] != "failed" else r.get("error", ""),
                     money))
    widths = [max(len(h), *(len(b[i]) for b in body)) for i, h in enumerate(head)]
    widths[3] = min(widths[3], 60)
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*head)] + [
        fmt.format(*[c if len(c) <= 60 else c[:57] + "..." for c in b]) for b in body]
    return "\n".join(lines)


def _cap_lines(exps: dict[str, Experiment], ests: dict[str, dict[str, Any]], n_runs: int,
               n_seeds: int, total_cap: float) -> list[str]:
    """The planned per-run caps, the experiment's total cap, and soft/hard gap warnings."""
    def money(v: float) -> str:
        return f"${v:g}" if v > 0 else "off"

    lines = []
    for a, exp in exps.items():
        b = exp.budget
        lines.append(f"caps: arm={a} per run soft={money(b.soft_usd)} hard={money(b.hard_usd)} "
                     f"measurement={money(b.measurement_usd)} x {n_seeds} seed(s)")
    hard_sum = sum(e.budget.hard_usd for e in exps.values()) * n_seeds
    if total_cap > 0:
        line = (f"caps: total ${total_cap:g} for the {n_runs} run(s) (budget.total_usd); a run "
                "starts only if spend so far + its hard_usd fits")
        if hard_sum > total_cap:
            line += (f"; hard ceilings sum to ${hard_sum:g}, so later runs may be skipped if "
                     "earlier ones spend near their ceilings")
        lines.append(line)
    else:
        lines.append("caps: no total cap (budget.total_usd: 0)"
                     + (f"; hard ceilings sum to ${hard_sum:g}" if hard_sum > 0 else ""))
    for a, exp in exps.items():
        b, per_round = exp.budget, float(ests[a].get("usd_per_round") or 0)
        if b.soft_usd > 0 and b.hard_usd > 0 and b.hard_usd - b.soft_usd < per_round:
            lines.append(
                f"warning: arm={a}: hard_usd - soft_usd = ${b.hard_usd - b.soft_usd:.4f} is less "
                f"than one round's estimated worst case ${per_round:.4f}. soft_usd is checked "
                "only between rounds, so a round can start below soft_usd, reach hard_usd "
                "mid-round and be discarded (its spend still counts). Leave at least one round "
                "between them (hard_usd >= soft_usd + one round).")
    return lines


def _confirm(prompt: str) -> None:
    try:
        ok = typer.confirm(prompt, default=False, err=True)
    except typer.Abort:
        ok = False
    if not ok:
        print("not confirmed; nothing ran (pass --yes to skip this question)", file=sys.stderr)
        raise typer.Exit(1)


@app.command()
def run(
    spec: Annotated[Path, typer.Argument(help="Experiment YAML.")],
    seed: Annotated[int | None, typer.Option(
        "--seed", help="Run only this seed (default: every seed in the YAML's `seeds:`).")] = None,
    arm: Annotated[str | None, typer.Option(
        "--arm", help="Run only this arm (default: every arm).")] = None,
    max_rounds: Annotated[int | None, typer.Option("--max-rounds", help="Override options.max_rounds.")] = None,
    out: Annotated[Path, typer.Option("--out", help="Parent directory for run dirs.")] = Path("runs"),
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask before spending.")] = False,
    rerun: Annotated[bool, typer.Option(
        "--rerun", help="Run again into runs/<id>__r<N> even if runs/<id> exists.")] = False,
    as_json: JsonOpt = False,
) -> None:
    """Run an experiment: every arm x every seed of the YAML, or the arm/seed you pick."""
    try:
        doc = load_experiment_yaml(spec)
        arms = [arm] if arm is not None else list(doc["arms"])
        seeds = [seed] if seed is not None else experiment_seeds(doc)
        exps = {a: _build(spec, a) for a in arms}
        for a, exp in exps.items():
            if max_rounds is None and "max_rounds" not in exp.options:
                raise SpecError(f"{spec}: arm {a!r}: no max_rounds in options; pass --max-rounds")
        ests = {a: _estimate(exp, seeds[0], max_rounds) for a, exp in exps.items()}
        total_cap = float(doc["budget"].get("total_usd") or 0)
        if total_cap > 0:
            unbounded = [a for a, e in exps.items() if e.budget.hard_usd <= 0]
            if unbounded:
                raise SpecError(f"{spec}: budget.total_usd ${total_cap:g} needs hard_usd > 0 on "
                                f"every arm that runs; arms without one: {unbounded}")
    except SpecError as e:
        _fail(e, 2, as_json)
    except Exception as e:  # noqa: BLE001
        _fail(e, 1, as_json)
    single = seed is not None and len(arms) == 1
    budgeted = any(_has_budget(e) for e in exps.values())
    n_runs = len(arms) * len(seeds)
    total = sum(est["usd"] for est in ests.values()) * len(seeds)
    if budgeted or any(est["llm_agents"] for est in ests.values()):
        for est in ests.values():
            _say(_estimate_line(est, None if single else len(seeds)), as_json)
        if not single:
            ceiling = sum(e.budget.hard_usd for e in exps.values()) * len(seeds)
            _say(f"estimate total: ${total:.4f} worst case over {n_runs} run(s)"
                 + (f"; hard ceilings sum to ${ceiling:.2f}" if ceiling > 0 else ""), as_json)
        if _only_fake(ests.values()):
            _say("  (fake: models only: prices are nominal, nothing is billed)", as_json)
    if budgeted:
        for line in _cap_lines(exps, ests, n_runs, len(seeds), total_cap):
            _say(line, as_json)
    if budgeted and not yes:
        _confirm(f"Start {n_runs} run(s), worst case ${total:.4f}?")

    rows: list[dict[str, Any]] = []
    capped: dict[str, Any] | None = None
    for a, sd in [(a, sd) for a in arms for sd in seeds]:
        exp = exps[a]
        rid = exp.run_id(sd)
        if capped is not None:
            capped["skipped"].append(rid)
            rows.append({"run_id": rid, "arm": a, "seed": sd, "outcome": "capped"})
            continue
        try:
            state, found = exp.existing(sd, max_rounds, out)
            if state == "same" and not rerun and found is not None:
                _say(f"{rid}: exists, skipping ({found.dir})", as_json)
                rows.append({**summary(found), "outcome": "skipped"})
                continue
            spent = sum(ledger_total(r.get("spend")) for r in rows)
            why = total_cap_refusal(spent, exp.budget.hard_usd, total_cap)
            if why:
                capped = {"reason": why, "total_usd": total_cap, "spent": spent,
                          "skipped": [rid]}
                _say(f"total cap: not starting {rid}: {why}", as_json)
                rows.append({"run_id": rid, "arm": a, "seed": sd, "outcome": "capped"})
                continue
            if state == "different" and not rerun:
                raise FileExistsError(
                    f"{out / rid} holds a run of a different configuration (spec_hash "
                    "differs); pass --rerun or another --out")
            target = Experiment.rerun_dir(out, rid) if state != "new" else None
            if not single:
                _say(f"{rid}: running" + (f" into {target}" if target else ""), as_json)
            r = exp.run(seed=sd, max_rounds=max_rounds, out=out, run_dir=target)
            rows.append({**summary(r), "outcome": "ran"})
        except Exception as e:  # noqa: BLE001 - one failed run does not stop the others
            msg = f"{type(e).__name__}: {e}"
            print(f"error: {rid}: {msg}", file=sys.stderr)
            rows.append({"run_id": rid, "arm": a, "seed": sd, "outcome": "failed",
                         "error": msg, "spec_error": isinstance(e, SpecError)})
    failed = [r for r in rows if r["outcome"] == "failed"]
    code = 0 if not failed else (2 if all(r["spec_error"] for r in failed) else 1)
    if capped is not None:
        code = code or 1
        print(f"error: total cap ${total_cap:g} reached; skipped {len(capped['skipped'])} "
              f"run(s): {', '.join(capped['skipped'])} ({capped['reason']})", file=sys.stderr)
    if single:
        r = rows[0]
        if r["outcome"] == "capped":
            if as_json:
                typer.echo(json.dumps({"ok": False, "error": f"total cap: {capped['reason']}",
                                       "exit_code": code, "capped": capped}))
            raise typer.Exit(code)
        if failed:
            if as_json:
                typer.echo(json.dumps({"ok": False, "error": r["error"], "exit_code": code}))
            raise typer.Exit(code)
        data = {k: v for k, v in r.items() if k != "outcome"}
        if r["outcome"] == "skipped":
            data["skipped"] = True
        typer.echo(json.dumps(data, default=str) if as_json else _human(data))
        return
    for r in rows:
        r.pop("spec_error", None)
    if as_json:
        typer.echo(json.dumps({"ok": not failed and capped is None, "exit_code": code,
                               "runs": rows, "capped": capped,
                               "estimate": {"arms": ests, "total_usd": total, "runs": n_runs,
                                            "total_cap_usd": total_cap}},
                              default=str))
    else:
        typer.echo(_table(rows))
    if code:
        raise typer.Exit(code)


def _estimate_table(ests: dict[str, dict[str, Any]], n_seeds: int, growth: bool) -> str:
    head = ["arm", "model agents", "rounds", "calls/turn", "calls/run", "probe calls", "per run",
            "seeds", "total"]
    if growth:
        head += ["per run +growth", "total +growth"]
    body = []
    for a, e in ests.items():
        cpt = e["calls_per_turn"]
        cpt = f"{min(cpt)}-{max(cpt)}" if isinstance(cpt, list) else str(cpt)
        row = [a, str(e["llm_agents"]), str(e["rounds"]), cpt, str(e["calls"]), str(e["probe_calls"]),
               _usd(e["usd_flat"]), str(n_seeds), _usd(e["usd_flat"] * n_seeds)]
        if growth:
            row += [_usd(e["usd"]), _usd(e["usd"] * n_seeds)]
        body.append(row)
    widths = [max(len(h), *(len(b[i]) for b in body)) for i, h in enumerate(head)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    return "\n".join([fmt.format(*head)] + [fmt.format(*b) for b in body])


@app.command()
def estimate(
    spec: Annotated[Path, typer.Argument(help="Experiment YAML.")],
    arm: Annotated[str | None, typer.Option(
        "--arm", help="Estimate only this arm (default: every arm).")] = None,
    seed: Annotated[int | None, typer.Option(
        "--seed", help="Estimate only this seed (default: every seed in the YAML's `seeds:`).")] = None,
    max_rounds: Annotated[int | None, typer.Option("--max-rounds", help="Override options.max_rounds.")] = None,
    calls_per_turn: Annotated[int | None, typer.Option(
        "--calls-per-turn",
        help="Model calls per turn to assume (default: options.max_calls_per_turn); a "
             "participant's own max_calls still caps it.")] = None,
    prompt_growth: Annotated[int, typer.Option(
        "--prompt-growth", metavar="TOKENS_PER_ROUND",
        help="Prompt tokens added per round (full-memory context growth); round r is priced at "
             "prompt_tokens + growth * (r - 1).")] = 0,
    as_json: JsonOpt = False,
) -> None:
    """Print a rough worst-case dollar estimate: every arm x every seed, or the arm/seed you pick
    (no run, no provider call)."""

    def go() -> dict[str, Any]:
        if prompt_growth < 0:
            raise SpecError("--prompt-growth must be >= 0")
        doc = load_experiment_yaml(spec)
        if arm is not None and arm not in doc["arms"]:
            raise SpecError(f"{spec}: unknown arm {arm!r}; available: {sorted(doc['arms'])}")
        arms = [arm] if arm is not None else list(doc["arms"])
        seeds = [seed] if seed is not None else experiment_seeds(doc)
        ests = {a: _estimate(_build(spec, a), seeds[0], max_rounds, calls_per_turn, prompt_growth)
                for a in arms}
        if seed is not None and len(arms) == 1:  # the single-run form: one estimate dict
            est = ests[arms[0]]
            if as_json:
                return est
            text = _estimate_line(est)[len("estimate: "):]
            if prompt_growth > 0:
                text += (f"\nflat (no growth): ${est['usd_flat']:.4f}; with prompt growth "
                         f"{prompt_growth} tokens/round: ${est['usd']:.4f}")
            return {"estimate": text}
        n_runs = len(arms) * len(seeds)
        total_flat = sum(e["usd_flat"] for e in ests.values()) * len(seeds)
        total = sum(e["usd"] for e in ests.values()) * len(seeds)
        data: dict[str, Any] = {"ok": True, "arms": ests, "seeds": seeds, "runs": n_runs,
                                "prompt_growth": prompt_growth, "total_usd": total,
                                "total_usd_flat": total_flat}
        if as_json:
            return data
        sources = sorted({e["calls_per_turn_source"] for e in ests.values()})
        lines = [_estimate_table(ests, len(seeds), prompt_growth > 0),
                 f"calls/turn from: {'; '.join(sources)}",
                 f"estimate total: ${total_flat:.4f} worst case over {n_runs} run(s)"]
        if prompt_growth > 0:
            lines.append(f"with prompt growth {prompt_growth} tokens/round: ${total:.4f} worst case")
        ceiling = sum(float(e["budget"].get("hard_usd") or 0) for e in ests.values()) * len(seeds)
        if ceiling > 0:
            lines.append(f"hard ceilings sum to ${ceiling:.2f}")
        if _only_fake(ests.values()):
            lines.append("(fake: models only: prices are nominal, nothing is billed)")
        return {"text": "\n".join(lines)}

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
    budget_soft: Annotated[float | None, typer.Option(
        "--budget-soft", help="New soft budget (USD) for the run's swarm spend in total, including "
                              "spend already recorded (discarded rounds included), not an "
                              "amount to add.")] = None,
    budget_hard: Annotated[float | None, typer.Option(
        "--budget-hard", help="New hard ceiling (USD) for the run's total spend, including spend "
                              "already recorded (discarded rounds included), not an amount to "
                              "add.")] = None,
    budget_measurement: Annotated[float | None, typer.Option(
        "--budget-measurement", help="New measurement budget (USD) for the run's total probe "
                                     "spend, including spend already recorded.")] = None,
    add_budget: Annotated[float | None, typer.Option(
        "--add-budget", help="Allow this many more USD from now: soft = swarm spend so far + X, "
                             "hard = total spend so far + X (each only if enforced).")] = None,
    as_json: JsonOpt = False,
) -> None:
    """Resume an interrupted (or budget-ended) run from its last committed round.

    The budget values are totals for the whole run; the current spend is printed before
    resuming (stderr under --json)."""

    def go() -> dict[str, Any]:
        r = Run(run_dir)
        changes = {k: v for k, v in (("soft_usd", budget_soft), ("hard_usd", budget_hard),
                                     ("measurement_usd", budget_measurement)) if v is not None}
        current = dict(r.meta.get("budget") or r.meta["spec"]["budget"])
        spend = r.spend
        if add_budget is not None:
            if add_budget <= 0:
                raise SpecError("--add-budget must be > 0")
            if {"soft_usd", "hard_usd"} & set(changes):
                raise SpecError("--add-budget cannot be combined with --budget-soft/--budget-hard")
            if not (current.get("soft_usd", 0) > 0 or current.get("hard_usd", 0) > 0):
                raise SpecError("--add-budget: the run has no soft or hard budget to raise; pass "
                                "--budget-hard X (the run's total)")
            if current.get("soft_usd", 0) > 0:
                changes["soft_usd"] = round(float(spend["swarm"]) + add_budget, 9)
            if current.get("hard_usd", 0) > 0:
                changes["hard_usd"] = round(ledger_total(spend) + add_budget, 9)
        _say(f"resume {r.id}: {_spend_text(spend)} so far; " + _budget_text(
            {k: v for k, v in {**current, **changes}.items() if k != "total_usd"}), as_json)
        budget = Budget(**{**current, **changes}) if changes else None
        data = summary(r.resume(budget=budget))
        data["budget"] = (budget or Budget(**current)).model_dump(mode="json")
        return data

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
    publish: Annotated[str | None, typer.Option(
        "--publish", help="Also upload view.html to this dataset repo (owner/name).")] = None,
    as_json: JsonOpt = False,
) -> None:
    """Build the self-contained replay page `view.html` for a run."""

    def go() -> dict[str, Any]:
        r = Run(run_dir)
        data = {**summary(r), "view": str(r.view())}
        if publish is not None:
            from .publish import publish_view

            res = publish_view(run_dir, publish)
            data["published"] = f"{res['url']}/blob/main/runs/{r.id}/view.html"
        return data

    _execute(go, as_json)


@app.command()
def models(
    provider: Annotated[str | None, typer.Option(
        "--provider", help="Only this provider prefix: hf or anthropic.")] = None,
    tools: Annotated[bool, typer.Option("--tools", help="Only models with native tool calling.")] = False,
    search: Annotated[str | None, typer.Option("--search", help="Substring of the model id.")] = None,
    refresh: Annotated[bool, typer.Option("--refresh", help="Refetch the HF router listing now.")] = False,
    as_json: JsonOpt = False,
) -> None:
    """List models you can put in a spec, who serves them, tool support and USD per M tokens."""

    def go() -> dict[str, Any]:
        from .providers import catalog

        if provider not in (None, "hf", "anthropic"):
            raise SpecError(f"--provider must be hf or anthropic, got {provider!r}")
        rows = catalog.entries(provider, tools=tools, search=search, refresh=refresh)
        if as_json:
            return {"ok": True, "models": [r.as_dict() for r in rows]}
        if not rows:
            hint = ("" if provider == "anthropic" else
                    f" (HF router listing unavailable or no match; cache: {catalog.cache_path()})")
            return {"text": "no models found" + hint}
        head = ("model (use in spec)", "tools", "context", "$/M in", "$/M out")

        def price(v: float | None) -> str:
            return f"{v:.2f}" if v is not None else "-"

        body = [(r.spec_id, {True: "yes", False: "no", None: "?"}[r.tools],
                 str(r.context_length or "-"), price(r.input), price(r.output)) for r in rows]
        widths = [max(len(h), *(len(b[i]) for b in body)) for i, h in enumerate(head)]
        fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        lines = [fmt.format(*head)] + [fmt.format(*b) for b in body]
        note = f"{len(rows)} model(s)."
        if any(r.provider == "hf" for r in rows):
            note += (" An hf id without ':served_by' lets the router pick the provider and is"
                     " priced at the most expensive listed one.")
        lines.append(note)
        return {"text": "\n".join(lines)}

    _execute(go, as_json)


@app.command()
def doctor(
    specs: Annotated[list[Path] | None, typer.Argument(
        help="Experiment YAMLs to check against (models, keys, extras).")] = None,
    offline: Annotated[bool, typer.Option("--offline", help="Skip the network checks.")] = False,
    as_json: JsonOpt = False,
) -> None:
    """Check Python, extras, API keys, provider reachability and the git checkout."""
    from . import doctor as doc

    results = doc.checks(specs or [], offline=offline)
    good = doc.ok(results)
    if as_json:
        typer.echo(json.dumps({"ok": good, "checks": [c.as_dict() for c in results]}))
    else:
        typer.echo(doc.report(results))
    if not good:
        raise typer.Exit(1)


@app.command()
def init(
    name: Annotated[str, typer.Argument(help="Experiment name; writes NAME.yaml and NAME.py.")] = "demo",
    directory: Annotated[Path, typer.Option("--dir", help="Where to write the files.")] = Path("."),
    force: Annotated[bool, typer.Option("--force", help="Overwrite existing files.")] = False,
    as_json: JsonOpt = False,
) -> None:
    """Write a starter experiment: NAME.yaml (two arms, fake LLM agents) and NAME.py (same, in Python)."""

    def go() -> dict[str, Any]:
        import re
        from importlib.resources import files

        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
            raise SpecError(f"name {name!r}: use letters, digits, '_' or '-', starting with a letter")
        targets = {directory / f"{name}.yaml": "starter.yaml.tmpl", directory / f"{name}.py": "starter.py.tmpl"}
        existing = [str(t) for t in targets if t.exists()]
        if existing and not force:
            raise FileExistsError(f"{', '.join(existing)} already exist(s); pass --force to overwrite")
        directory.mkdir(parents=True, exist_ok=True)
        for target, template in targets.items():
            text = (files("swarmlab") / "templates" / template).read_text()
            target.write_text(text.replace("__NAME__", name))
        yaml_path = directory / f"{name}.yaml"
        return {"ok": True, "yaml": str(yaml_path), "python": str(directory / f"{name}.py"),
                "text": (f"wrote {yaml_path} and {directory / f'{name}.py'}\n"
                         f"next: swarmlab run {yaml_path}")}

    _execute(go, as_json)


# ---- HF Jobs (WP8) -----------------------------------------------------------------------------
job_app = typer.Typer(
    name="job", no_args_is_help=True,
    help="Run a spec in an HF Job next to a vLLM server; check, follow and fetch it.")
app.add_typer(job_app, name="job")


def _seed_list(text: str | None) -> list[int] | None:
    if text is None:
        return None
    try:
        return [int(x) for x in text.replace(" ", "").split(",") if x]
    except ValueError as e:
        raise SpecError(f"--seeds must be comma-separated integers, got {text!r}") from e


@job_app.command("run")
def job_run(
    spec: Annotated[Path, typer.Argument(help="Experiment YAML whose models are vllm:<model>.")],
    model: Annotated[str, typer.Option("--model", help="HF model id vLLM serves, e.g. Qwen/Qwen3.5-9B.")],
    flavor: Annotated[str, typer.Option("--flavor", help="HF Jobs hardware flavor.")] = "a100-large",
    arm: Annotated[list[str] | None, typer.Option(
        "--arm", help="Arm to run (repeatable; default: every arm).")] = None,
    seeds: Annotated[str | None, typer.Option(
        "--seeds", help="Comma-separated seeds (default: the YAML's seeds).")] = None,
    timeout: Annotated[str | None, typer.Option(
        "--timeout", help="Job timeout, e.g. 2h (default: 1.5x the estimate).")] = None,
    bucket: Annotated[str, typer.Option("--bucket", help="Bucket for stage files and runs.")] = (
        "hf://buckets/cmpatino/swarmlab-runs"),
    image: Annotated[str | None, typer.Option("--image", help="Docker image (default: uv python3.12).")] = None,
    vllm_version: Annotated[str | None, typer.Option("--vllm-version", help="vLLM pip version.")] = None,
    max_model_len: Annotated[int | None, typer.Option("--max-model-len", help="vLLM --max-model-len.")] = None,
    revision: Annotated[str | None, typer.Option("--revision", help="Model revision to pin.")] = None,
    per_round: Annotated[float | None, typer.Option(
        "--per-round", help="Seconds per round for the estimate (default 30 + 2*N).")] = None,
    stage_dir: Annotated[Path | None, typer.Option(
        "--stage-dir", help="Also write the stage files (spec, bootstrap, plan, wheel) here.")] = None,
    launch: Annotated[bool, typer.Option(
        "--launch", help="Upload the stage dir and submit the job (otherwise only print).")] = False,
    allow_dirty: Annotated[bool, typer.Option(
        "--allow-dirty", help="Launch from a checkout with uncommitted changes.")] = False,
    ledger: Annotated[Path, typer.Option("--ledger", help="Local launch ledger (JSONL).")] = Path(
        "runs/jobs.jsonl"),
    as_json: JsonOpt = False,
) -> None:
    """Print (and with --launch submit) an HF Job: vLLM serving MODEL + `swarmlab run` per arm x seed."""
    from .jobs import launch as jl

    def go() -> dict[str, Any]:
        kw: dict[str, Any] = {}
        for key, val in (("image", image), ("vllm_version", vllm_version),
                         ("max_model_len", max_model_len), ("revision", revision),
                         ("per_round_s", per_round), ("timeout", timeout)):
            if val is not None:
                kw[key] = val
        plan = jl.plan_job(spec, model=model, flavor=flavor, arms=arm or None,
                           seeds=_seed_list(seeds), bucket=bucket, **kw)
        data: dict[str, Any] = {"ok": True, "plan": plan.as_dict(), "launched": False}
        text = jl.describe(plan)
        if stage_dir is not None:
            jl.stage(plan, stage_dir)
            data["stage_dir"] = str(stage_dir)
            text += f"\nstaged locally: {stage_dir}"
        if not launch:
            text += "\nnot launched (pass --launch to upload the stage dir and submit)"
        else:
            if plan.dirty and not allow_dirty:
                raise SpecError("the checkout has uncommitted changes; commit first or pass "
                                "--allow-dirty")
            row = jl.submit(plan, hf=jl.which_hf(), ledger=ledger)
            data.update(launched=True, job=row)
            text += (f"\nlaunched job {row['job_id']}  {row['url']}\n"
                     f"follow: swarmlab job logs {row['job_id']} --follow\n"
                     f"status: swarmlab job status {row['job_id']}\n"
                     f"fetch:  swarmlab job fetch <run_id> --tag {plan.tag}")
        data["text"] = text
        return data if as_json else {"text": text}

    _execute(go, as_json)


@job_app.command("status")
def job_status(
    job_id: Annotated[str, typer.Argument(help="HF Job id.")],
    as_json: JsonOpt = False,
) -> None:
    """Stage, flavor, running time and compute cost so far of a job."""
    from .jobs import remote

    def go() -> dict[str, Any]:
        st = remote.status(job_id)
        if as_json:
            return {"ok": True, **st}
        cost = f"${st['cost_usd']:.2f}" if st["cost_usd"] is not None else "-"
        return {"text": (f"job {st['job_id']}  stage={st['stage']}  flavor={st['flavor']}  "
                         f"running={st['running_secs'] or 0}s  compute={cost}  tag={st['tag']}"
                         + (f"\n  message: {st['message']}" if st["message"] else "")
                         + (f"\n  {st['url']}" if st["url"] else ""))}

    _execute(go, as_json)


@job_app.command("logs")
def job_logs(
    job_id: Annotated[str, typer.Argument(help="HF Job id.")],
    follow: Annotated[bool, typer.Option("--follow", "-f", help="Stream until the job ends.")] = False,
) -> None:
    """Print a job's logs (`hf jobs logs`)."""
    from .jobs import remote

    code = remote.logs(job_id, follow)
    if code:
        raise typer.Exit(code)


@job_app.command("fetch")
def job_fetch(
    run_id: Annotated[str, typer.Argument(help="Run id, or <tag>/<run id>.")],
    out: Annotated[Path, typer.Option("--out", help="Parent directory for the run dir.")] = Path("runs"),
    bucket: Annotated[str, typer.Option("--bucket", help="Bucket (or a local dir with the same layout).")] = (
        "hf://buckets/cmpatino/swarmlab-runs"),
    tag: Annotated[str | None, typer.Option("--tag", help="Job tag (default: newest holding the run).")] = None,
    as_json: JsonOpt = False,
) -> None:
    """Download a run dir from the bucket so `Run.load`, `view` and `fork` work locally."""
    from .jobs import remote

    def go() -> dict[str, Any]:
        d = remote.fetch(run_id, out, bucket, tag)
        return {**summary(Run(d)), "fetched": str(d)}

    _execute(go, as_json)


# ---- M4: export, publish, report, prompts (docs/INTERFACE-M4.md) ------------------------------
@app.command("export")
def export_cmd(
    run_dir: Annotated[Path, typer.Argument(help="Run directory.")],
    out: Annotated[Path | None, typer.Option("--out", help="Export directory (default RUN_DIR/export).")] = None,
    as_json: JsonOpt = False,
) -> None:
    """Export a run: Parquet tables per event family, pi-format sessions, raw copies."""
    from .export import export_run

    def go() -> dict[str, Any]:
        d = export_run(run_dir, out)
        doc = json.loads((d / "run.json").read_text())
        tables = {f: t["rows"] for f, t in doc["tables"].items()}
        data = {"ok": True, "run_id": doc["run_id"], "export": str(d), "tables": tables,
                "sessions": len(doc["sessions"]), "blobs": doc["blobs"]["included"]}
        if as_json:
            return data
        rows = ", ".join(f"{f}={n}" for f, n in tables.items())
        return {"text": (f"exported {doc['run_id']} -> {d}\n  rows: {rows}\n"
                         f"  sessions: {len(doc['sessions'])}  raw blobs: {doc['blobs']['included']}")}

    _execute(go, as_json)


@app.command("publish")
def publish_cmd(
    source: Annotated[Path, typer.Argument(help="A run directory or a directory of runs.")],
    repo: Annotated[str | None, typer.Option(
        "--repo", help="Dataset repo owner/name (default: <your namespace>/<experiment>).")] = None,
    public: Annotated[bool, typer.Option(
        "--public", help="Make the repo public and tag it format:agent-traces.")] = False,
    tag: Annotated[list[str] | None, typer.Option("--tag", help="Extra card tag (repeatable).")] = None,
    as_json: JsonOpt = False,
) -> None:
    """Export (if needed) and upload finished runs to a private Hub dataset repo per experiment."""
    from .publish import PublishError, publish

    def go() -> dict[str, Any]:
        try:
            res = publish(source, repo, public=public, tag=tag or [])
        except PublishError as e:
            raise SpecError(str(e)) from e
        if as_json:
            return res
        lines = [f"{r['url']}  {'public' if r['public'] else 'private'}  runs={len(r['runs'])}  "
                 f"uploaded={r['uploaded']} unchanged={r['unchanged']}" for r in res["repos"]]
        if res["skipped"]:
            lines.append("skipped: " + ", ".join(res["skipped"]))
        return {"text": "\n".join(lines)}

    _execute(go, as_json)


@app.command("fetch-published")
def fetch_published_cmd(
    repo: Annotated[str, typer.Argument(help="Dataset repo owner/name.")],
    run_id: Annotated[str, typer.Argument(help="Run id (see the repo's index.json).")],
    out: Annotated[Path, typer.Option("--out", help="Parent directory for the run dir.")] = Path("runs"),
    force: Annotated[bool, typer.Option("--force", help="Overwrite an existing run dir.")] = False,
    as_json: JsonOpt = False,
) -> None:
    """Rebuild a run dir from a published run so `Run.load`, `replay`, `view` and `fork` work."""
    from .publish import PublishError, fetch_published

    def go() -> dict[str, Any]:
        try:
            d = fetch_published(repo, run_id, out, force=force)
        except PublishError as e:
            raise SpecError(str(e)) from e
        return {**summary(Run(d)), "fetched": str(d)}

    _execute(go, as_json)


@app.command("report")
def report_cmd(
    runs_dir: Annotated[Path, typer.Argument(help="Directory of run directories.")],
    out: Annotated[Path | None, typer.Option("--out", help="Also write the Markdown here.")] = None,
    title: Annotated[str, typer.Option("--title", help="Report title.")] = "swarmlab report",
    as_json: JsonOpt = False,
) -> None:
    """Markdown report over the finished runs in RUNS_DIR (accuracy, consensus, reading, health)."""
    from .report import write_report

    def go() -> dict[str, Any]:
        if not runs_dir.is_dir():
            raise SpecError(f"{runs_dir} is not a directory")
        text = write_report(runs_dir, out, title)
        if as_json:
            return {"ok": True, "report": text, "out": str(out) if out else None}
        return {"text": text}

    _execute(go, as_json)


@app.command("prompts")
def prompts_cmd(
    spec: Annotated[Path, typer.Argument(help="Experiment YAML.")],
    arm: Annotated[str | None, typer.Option("--arm", help="Arm (required when the YAML has several).")] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Seed (default: the YAML's first seed).")] = None,
    as_json: JsonOpt = False,
) -> None:
    """Render the system prompt and round-1 user message for one agent per participant group.

    No model is called; use it for the second-agent review of what each arm sees."""
    from .prompts_cmd import prompts_text, render_prompts

    def go() -> dict[str, Any]:
        exp = _build(spec, arm)
        s = seed if seed is not None else experiment_seeds(load_experiment_yaml(spec))[0]
        rows = render_prompts(exp, s)
        if as_json:
            return {"ok": True, "arm": exp.arm, "seed": s, "groups": rows}
        return {"text": prompts_text(rows, exp.arm, s)}

    _execute(go, as_json)


def main() -> None:  # pragma: no cover - `python -m swarmlab.cli`
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
