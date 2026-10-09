"""`swarmlab report RUNS_DIR [--out FILE] [--title T]`: a Markdown report over finished runs.

Generalised from `tools/m2_report.py` (the M2 phase-1 Flag Game report); on the archived M2 runs
with `--title "M2 phase-1 Flag Game report"` it reproduces
`docs/notes/m2-phase1-report-2026-10-06.md` byte for byte apart from the rows and lines added
since (`post_rate` trajectory rows, "probes skipped" lines, the tool-health `max_tokens` and `rejected tool calls` columns). Read-only over the runs dir.

Decisions:

- Every run directory directly under RUNS_DIR whose `run.json` says `status: ended` is loaded with
  `Run.load` (a full replay: a run that does not replay is listed as skipped with the error).
  Runs are grouped by arm (the middle part of `<experiment>__<arm>__s<seed>`, else `run.json`'s
  arm, else `?`).
- Per-round columns span rounds 1..max(last committed round) over the runs (10 for M2), and the
  summary's read rate averages rounds 2..that maximum.
- Sections: summary, trajectories (the belief and read-rate rows `accuracy`, `consensus`,
  `entropy`, `read_rate` when some run logs them, `n/a` where a run does not; then every other
  logged metric under its full dotted name, so a custom world's metrics show up; plus
  `post_rate`, the share of the round's turns that posted, computed from the events so it is
  there whether or not `comm.post_rate` was logged), reading behaviour, tool-protocol health and
  protocol health for every world; "where the swarm went" and "probe vs world belief" only when
  a run's world is a FlagGame (or subclass) or some run has committed `guess` actions (truth and
  rival come from rebuilding the world, since `verify()` is not persisted); "Coloring (final
  grid)" (the recorded `score()` per run) only when a run's world is a ColoringGrid.
- M6 "Terminal states (Flag Game paper)" (FlagGame-style runs, after "where the swarm went"):
  per arm, the share of runs whose endpoint is correct consensus / wrong consensus / polarized /
  fragmented (`metrics.belief.classify_state`, thresholds 0.85 / 0.25) and the mean terminal
  truth mass. The endpoint is the manager's final decision when a blind agent guessed (manager
  protocol: correct or wrong consensus of one), else the last belief-probe round's answers of the
  sighted agents (pairwise: the paper's probes), else their final committed guesses. Runs with an
  empty endpoint are left out (`runs` counts the classified ones). Guesses are read by their
  canonical name (`feedback["candidate"]` when the world gives one).
- Protocol health (per arm, any world; the first thing to check after a paid run): turn end
  kinds, turns errored with the first error (for a traceback: the first line of its final
  exception), finish reasons of all inference responses, responses and cache hits, retries
  (responses with `attempts > 1` and the extra attempts), latency of uncached responses (median,
  nearest-rank p90, max), swarm model calls per turn (turns without a call count as 0), cost per
  committed round (uncached responses, swarm + measurement), tool calls rejected/answered with
  the rejection reasons (the part of `error` before `:`), world actions not accepted/committed
  with the feedback `error`, probes skipped by reason, turns refused (final model response a
  refusal, `runner.refusal_of_turn`) and refused responses (probes included; cache hits count,
  as in finish reasons), each by category (`none` when the provider named none).
- Probe vs world belief: a skipped probe (`parsed.skipped`: no probe context, a budget, a
  provider error) is left out of that round's probe columns rather than counted as no answer,
  and each arm's table is followed by a `Probes skipped:` line with the counts by reason.
- Tool protocol health counts truncated responses (`inference_response.finish_reason`) in two
  columns: `length` (OpenAI-compatible providers) and `max_tokens` (Anthropic); both mean the
  reply hit `max_tokens`. (`max_tokens` is the last column so the M2 table keeps its layout.)
  `rejected tool calls` (appended after it) is `rejected/answered`: `tool_returned` events with
  `ok=False` (a world refusal such as `paints_per_round`, `turn_ended`, `not_allowed`, bad args)
  over all of them, both without the call that hit the turn cap (counted as a `cap` yield). A
  high share alongside many `cap` yields means agents were spinning, not busy.
- The spend/run column is the ledger (discarded rounds included). When some summarised run
  charged spend in discarded rounds (`export.discarded_spend`), a line under the summary gives
  the ledger total and the discarded part, so it reconciles with exports; otherwise the report
  is unchanged (the M2 reproduction stays byte-identical).
- Simulated runs (every participant model and probe coder model is `fake:`; scripted-only runs
  are not simulated) are named in a line under the header and left out of all tables and the
  spend total; `include_fake=True` (CLI `--include-fake`) puts them in the tables as arm
  `<arm> (simulated)`, still outside the spend total.
"""
from __future__ import annotations

import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from .budget import ledger_total
from .experiment import Experiment, Run
from .export import discarded_spend
from .rng import derive
from .runner import refusal_of_turn

DEFAULT_TITLE = "swarmlab report"

NA = "n/a"


def f(x, p=3):
    return NA if x is None else f"{x:.{p}f}"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def sd(xs):
    xs = [x for x in xs if x is not None]
    return st.stdev(xs) if len(xs) > 1 else (0.0 if xs else None)


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def mseries(run, name):
    try:
        return {r: v for r, v, _ in run.metrics.get(name, [])}
    except Exception:  # noqa: BLE001 - a broken run is reported as missing data
        return {}


def scan(run):
    """One pass over the typed events: everything the report needs."""
    d = {"guess_by_round": {}, "turns": Counter(), "reads": Counter(), "read_deliv": Counter(),
         "read_turns": Counter(), "agents_read": set(), "agents": set(), "yields": Counter(),
         "finish": Counter(), "err": 0, "cached": 0, "nresp": 0, "posts": Counter(), "t0": None,
         "t1": None, "turn_n": Counter(), "length": 0, "max_tokens": 0, "posters": defaultdict(set),
         "tool_returns": 0, "rejected_calls": 0,
         # protocol health (any world)
         "resp_finish": Counter(), "attempts": 0, "retried": 0, "latency": [],
         "calls_per_turn": Counter(), "cost_by_round": Counter(), "reject_errors": Counter(),
         "probe_skips": Counter(), "probes": 0, "errors": [], "metric_series": defaultdict(dict),
         "committed": 0, "not_accepted": Counter(), "turn_refusals": Counter(),
         "resp_refusals": Counter()}
    cur = {}
    turn_keys = []
    last_round = 0
    for ev in run.events_all:
        t, r, a = ev.type, ev.round, ev.agent
        if t == "run_started":
            d["t0"] = ev.ts
        elif t == "run_ended":
            d["t1"] = ev.ts
        elif t == "turn_ended":
            d["turns"][r] += 1
            turn_keys.append((r, a))
            d["agents"].add(a)
            d["yields"][ev.yield_kind] += 1
            if ev.yield_kind == "error" or getattr(ev, "error", None):
                d["err"] += 1
                d["errors"].append((r, a, first_error_line(getattr(ev, "error", None))))
            for fr in (ev.usage or {}).get("finish_reasons", []) if isinstance(ev.usage, dict) else []:
                d["finish"][fr] += 1
            category = refusal_of_turn(ev.usage)
            if category is not None:
                d["turn_refusals"][category] += 1
        elif t == "tool_returned":
            res = ev.result if isinstance(ev.result, dict) else {}
            if res.get("error") != "cap":  # the capping call is counted by the `cap` yield kind
                d["tool_returns"] += 1
                d["rejected_calls"] += not res.get("ok", True)
                if not res.get("ok", True):
                    d["reject_errors"][str(res.get("error") or "?").split(":")[0][:40]] += 1
        elif t == "read":
            d["reads"][r] += 1
            d["read_deliv"][r] += len(ev.delivery_ids)
            d["agents_read"].add(a)
        elif t == "post":
            d["posts"][r] += 1
            d["posters"][r].add(a)
        elif t == "inference_response":
            d["nresp"] += 1
            d["cached"] += bool(getattr(ev, "cached", False))
            fr = getattr(ev, "finish_reason", None)
            if fr == "length":  # OpenAI-compatible providers
                d["length"] += 1
            elif fr == "max_tokens":  # Anthropic
                d["max_tokens"] += 1
            d["resp_finish"][fr or "?"] += 1
            if fr == "refusal":
                d["resp_refusals"][getattr(ev, "refusal_category", None) or "none"] += 1
            n_att = getattr(ev, "attempts", 1) or 1
            d["attempts"] += n_att
            d["retried"] += n_att > 1
            if not getattr(ev, "cached", False):
                d["latency"].append(float(getattr(ev, "latency_s", 0.0) or 0.0))
                d["cost_by_round"][r] += float(getattr(ev, "cost_usd", 0.0) or 0.0)
        elif t == "inference_attempt":
            if getattr(ev, "category", "swarm") == "swarm":
                d["calls_per_turn"][(r, a)] += 1
        elif t == "probe":
            d["probes"] += 1
            parsed = getattr(ev, "parsed", None)
            if isinstance(parsed, dict) and "skipped" in parsed:
                d["probe_skips"][str(parsed["skipped"])] += 1
        elif t == "metric":
            d["metric_series"][ev.name][r] = ev.value
        elif t == "action_committed":
            act = ev.action if isinstance(ev.action, dict) else dict(ev.action)
            if act.get("name") == "guess" and ev.accepted:
                fb = ev.feedback if isinstance(ev.feedback, dict) else {}
                canonical = fb.get("candidate")  # M6: the world's canonical name (real flags)
                cur[str(a)] = canonical if isinstance(canonical, str) else act["args"]["candidate"]
            d["committed"] += 1
            if not ev.accepted:
                fb = ev.feedback if isinstance(ev.feedback, dict) else {}
                d["not_accepted"][str(fb.get("error") or "?").split(":")[0][:40]] += 1
        elif t == "round_committed":
            d["guess_by_round"][r] = dict(cur)
            last_round = r
    d["last_round"] = last_round
    d["turn_keys"] = turn_keys
    d["wall"] = (d["t1"] - d["t0"]) if d["t0"] and d["t1"] else None
    return d


def first_error_line(text) -> str:
    """The line of a turn error worth quoting: for a traceback, the first line of its final
    exception (`RuntimeError: ...`); else the first non-empty line."""
    raw = str(text or "").splitlines()
    frames = [i for i, ln in enumerate(raw) if ln.startswith("  File ")]
    tail = raw[frames[-1] + 1:] if frames else raw
    line = next((ln.strip() for ln in tail if ln.strip() and (not frames or not ln[0].isspace())),
                None)
    if line is None:
        line = next((ln.strip() for ln in raw if ln.strip()), "(no error text)")
    return line if len(line) <= 200 else line[:199] + "…"


def pct(xs, q):
    """Nearest-rank percentile of `xs` (None when empty)."""
    xs = sorted(xs)
    if not xs:
        return None
    return xs[min(len(xs) - 1, max(0, -(-len(xs) * q // 100) - 1))]


def counts(c: Counter) -> str:
    return ", ".join(f"{k} {v}" for k, v in sorted(c.items(), key=lambda kv: (-kv[1], str(kv[0])))) or "none"


def health_rows(rs) -> list[list]:
    """The per-arm protocol-health table (any world): what to check first after a paid run."""
    ds = [d for _, d in rs]
    turns = sum(sum(d["turns"].values()) for d in ds)
    yields, finish, rej, skips, refused = Counter(), Counter(), Counter(), Counter(), Counter()
    turn_ref, resp_ref = Counter(), Counter()
    for d in ds:
        turn_ref.update(d["turn_refusals"])
        resp_ref.update(d["resp_refusals"])
        refused.update(d["not_accepted"])
        yields.update(d["yields"])
        finish.update(d["resp_finish"])
        rej.update(d["reject_errors"])
        skips.update(d["probe_skips"])
    nresp = sum(d["nresp"] for d in ds)
    lat = [x for d in ds for x in d["latency"]]
    # model calls per turn: every turn counts, also those that made no call
    cpt = [d["calls_per_turn"].get((r, a), 0) for d in ds for r, a in d["turn_keys"]]
    cost = [d["cost_by_round"].get(k, 0.0) for d in ds for k in range(1, d["last_round"] + 1)]
    errs = [(run.id, *e) for run, d in rs for e in d["errors"]]
    answered = sum(d["tool_returns"] for d in ds)
    rejected = sum(d["rejected_calls"] for d in ds)
    nprobe = sum(d["probes"] for d in ds)
    err_text = str(len(errs)) + (f" of {turns}" if turns else "")
    if errs:
        rid, r, a, line = errs[0]
        err_text += f"; first: `{rid}` r{r} {a}: {line}"
    rows = [
        ["runs / turns", f"{len(rs)} / {turns}"],
        ["turn end kinds", counts(yields)],
        ["turns errored", err_text],
        ["turns refused", (f"{sum(turn_ref.values())} of {turns} ({counts(turn_ref)})"
                           if turn_ref else "0")],
        ["refused responses, probes included", (f"{sum(resp_ref.values())} ({counts(resp_ref)})"
                                                if resp_ref else "0")],
        ["finish reasons (responses)", counts(finish)],
        ["inference responses, probes included (all / cache hits)", f"{nresp} / {sum(d['cached'] for d in ds)}"],
        ["retries (responses retried / extra attempts)",
         f"{sum(d['retried'] for d in ds)} / {sum(d['attempts'] for d in ds) - nresp}"],
        ["latency s, uncached (median / p90 / max)",
         " / ".join(f(x, 2) for x in (st.median(lat) if lat else None, pct(lat, 90),
                                      max(lat) if lat else None))],
        ["model calls per turn, swarm (mean / max)", f"{f(mean(cpt), 2)} / {max(cpt) if cpt else NA}"],
        ["cost per round, swarm + measurement (mean / max)",
         f"${f(mean(cost))} / ${f(max(cost) if cost else None)}"],
        ["tool calls rejected / answered",
         f"{rejected} / {answered}" + (f" ({counts(rej)})" if rej else "")],
        ["world actions not accepted / committed",
         f"{sum(refused.values())} / {sum(d['committed'] for d in ds)}"
         + (f" ({counts(refused)})" if refused else "")],
        ["probes skipped", (f"{sum(skips.values())} of {nprobe} ({counts(skips)})" if skips else
                            ("none" if nprobe else "no probes"))],
    ]
    return rows


TRAJECTORY_ROWS = [("accuracy", "belief.accuracy"), ("consensus", "belief.consensus"),
                   ("entropy", "belief.entropy"), ("read_rate", "comm.read_rate")]
COLORING_KEYS = ("coverage", "correct", "wrong", "unpainted", "paints", "useful_paints",
                 "duplicate_paints", "wrong_paints", "overwritten_paints", "wasted_paints")


def world_kind(run) -> str:
    """`flaggame`, `coloring` (built-ins and their subclasses) or `other`."""
    from .registry import resolve
    from .world.coloring import ColoringGrid
    from .world.flaggame import FlagGame

    try:
        cls = resolve(run.spec.world.type, "swarmlab.worlds")
    except Exception:  # noqa: BLE001 - a custom world that does not import here
        return "other"
    if isinstance(cls, type) and issubclass(cls, FlagGame):
        return "flaggame"
    if isinstance(cls, type) and issubclass(cls, ColoringGrid):
        return "coloring"
    return "other"


def coloring_section(arms) -> list[str]:
    """Final grid score per run (ColoringGrid `score()` as recorded in run.json)."""
    rows = []
    for arm, rs in sorted(arms.items()):
        for run, d in rs:
            if world_kind(run) != "coloring":
                continue
            sc = run.score or {}
            rows.append([run.id, d["last_round"]] +
                        [f(sc.get(k)) if isinstance(sc.get(k), float) else sc.get(k, NA)
                         for k in COLORING_KEYS])
    return ["## Coloring (final grid)", ""] + table(["run", "rounds", *COLORING_KEYS], rows) + [""]


def world_verify(run, agents):
    """verify() is not persisted: rebuild the world exactly as the runner does."""
    seed = run.spec.options.seed
    exp = Experiment.from_spec(run.spec)
    ag = [p.agent for p in exp.participants if getattr(p, "agent", None)] or sorted(agents)
    exp.world.reset(derive(seed, "world"), ag)
    return exp.world.verify()


def truth_info(run, agents):
    v = world_verify(run, agents)
    rec = run.score.get("truth")
    return v["truth"], v["rival"], v["truth"] == rec if rec is not None else None


def terminal_distribution(run, d, verify) -> tuple[Counter, str]:
    """M6: the run's endpoint distribution and its kind: the blind agents' final guesses
    (`manager`) when a blind agent guessed, else the last belief-probe round's answers
    (`probes`), else the final committed guesses of the sighted agents (`guesses`)."""
    final = d["guess_by_round"].get(d["last_round"], {})
    blind = set(verify.get("blind") or [])
    if blind and any(a in final for a in blind):
        return Counter(v for a, v in final.items() if a in blind), "manager"
    try:
        pr = run.probes.get("belief", [])
    except Exception:  # noqa: BLE001 - a broken run is reported as missing data
        pr = []
    answered = [(r, ag, parsed) for r, ag, parsed, ok in pr
                if ok and isinstance(parsed, dict) and isinstance(parsed.get("candidate"), str)]
    if answered:
        last = max(r for r, _, _ in answered)
        return Counter(p["candidate"] for r, ag, p in answered if r == last and ag not in blind), "probes"
    return Counter(v for a, v in final.items() if a not in blind), "guesses"


def terminal_section(arms) -> list[str]:
    """M6 §5: the paper's terminal-state shares per arm across seeds, and terminal truth mass."""
    from .metrics.belief import STATES, classify_state

    rows = []
    for arm, rs in sorted(arms.items()):
        states: Counter = Counter()
        mass, kinds, n = [], Counter(), 0
        for run, d in rs:
            try:
                v = world_verify(run, d["agents"])
            except Exception:  # noqa: BLE001, S112 - no verifiable truth: not classified
                continue
            dist, kind = terminal_distribution(run, d, v)
            label = classify_state(dist, v.get("truth"))
            if label is None:
                continue
            n += 1
            states[label] += 1
            kinds[kind] += 1
            mass.append(dist.get(v.get("truth"), 0) / sum(dist.values()))
        if not n:
            rows.append([arm, 0] + [NA] * (len(STATES) + 2))
            continue
        rows.append([arm, n] + [f(states[s] / n, 2) for s in STATES]
                    + [f(mean(mass), 3), ", ".join(f"{k} {c}" for k, c in sorted(kinds.items()))])
    head = ["arm", "runs", "correct consensus", "wrong consensus", "polarized", "fragmented",
            "terminal truth mass", "endpoint"]
    note = ("Share of runs per class of the endpoint distribution (s1 >= 0.85 consensus, camps "
            ">= 0.25); terminal truth mass is the mean share of the endpoint on the truth. "
            "Endpoint: the manager's final decision when a blind agent guessed, else the last "
            "belief-probe round, else the final committed guesses.")
    return ["## Terminal states (Flag Game paper)", "", note, ""] + table(head, rows) + [""]


def is_simulated(meta: dict) -> bool:
    """True when every model the run calls (participants, probe coders) is a `fake:` model."""
    spec = meta.get("spec") or {}
    models = [(p.get("params") or {}).get("model") for p in spec.get("participants") or []]
    models += [(p.get("params") or {}).get("coder_model") for p in spec.get("probes") or []]
    models = [m for m in models if isinstance(m, str) and ":" in m]
    return bool(models) and all(m.startswith("fake:") for m in models)


def spend_lines(runs, n_simulated: int = 0) -> list[str]:
    """Total ledger spend of the real (non-simulated) runs summarised, when some of it was charged
    in discarded rounds or simulated runs are around; otherwise the spend/run column already
    adds up to the ledger and nothing is added."""
    discarded = sum(discarded_spend(r.dir) for r in runs)
    if discarded <= 0 and not n_simulated:
        return []
    total = sum(ledger_total(r.spend) for r in runs)
    line = f"Total spend (ledger, swarm + measurement) over the {len(runs)} real run(s): ${total:.3f}"
    if discarded > 0:
        line += (f", of which ${discarded:.3f} was charged in discarded rounds (hard-ceiling "
                 "aborts, not in the logged rounds; `swarmlab export` puts them in "
                 "`tables/discarded_inference`)")
    if n_simulated:
        line += f"; {n_simulated} simulated run(s) (nominal `fake:` prices) not counted"
    return [line + ".", ""]


def build_report(runs_dir: Path | str, title: str = DEFAULT_TITLE, include_fake: bool = False) -> str:
    """The report for every ended run directly under `runs_dir`, as Markdown.

    Simulated runs (`is_simulated`: only `fake:` models) are listed but left out of the summary
    and every section unless `include_fake`, which adds them under arm `<arm> (simulated)`; they
    never count towards the spend total."""
    arms = defaultdict(list)
    skipped = []
    simulated = []
    for d in sorted(Path(runs_dir).iterdir()):
        rj = d / "run.json"
        if not rj.is_file():
            continue
        try:
            meta = json.loads(rj.read_text())
            if meta.get("status") != "ended":
                skipped.append(d.name)
                continue
            fake = is_simulated(meta)
            if fake:
                simulated.append(d.name)
                if not include_fake:
                    continue
            run = Run.load(d)
        except Exception as e:  # noqa: BLE001
            skipped.append(f"{d.name} ({type(e).__name__})")
            continue
        parts = run.id.split("__")
        arm = parts[1] if len(parts) >= 3 else (run.meta.get("arm") or "?")
        arms[f"{arm} (simulated)" if fake else arm].append((run, scan(run)))
    last = [d["last_round"] for rs in arms.values() for _, d in rs]
    R = range(1, max([*last, 1]) + 1)
    kinds = {world_kind(run) for rs in arms.values() for run, _ in rs}
    flag = "flaggame" in kinds or any(d["guess_by_round"].get(d["last_round"])
                                      for rs in arms.values() for _, d in rs)
    L = [f"# {title}", "", f"Runs dir: `{runs_dir}`; skipped (not ended / unreadable): {', '.join(skipped) or 'none'}", ""]
    if simulated:
        how = ("included below as `<arm> (simulated)`" if include_fake else
               "left out of every table below (`--include-fake` adds them)")
        L += [f"Simulated runs (only `fake:` models, nominal prices; {how}): {', '.join(simulated)}", ""]

    # summary
    rows = []
    for arm, rs in sorted(arms.items()):
        fin = lambda n, rs=rs: [(mseries(r, n) or {}).get(max(mseries(r, n) or [0])) for r, _ in rs]
        acc, con = fin("belief.accuracy"), fin("belief.consensus")
        rt = []
        for r, _ in rs:
            c = mseries(r, "belief.consensus")
            rt.append(next((k for k in sorted(c) if c[k] is not None and c[k] >= 0.9), None))
        reached = [x for x in rt if x is not None]
        med = (str(st.median(reached)) if len(reached) * 2 > len(rt) else "never") if rt else NA
        rr = mean([mean([v for k, v in mseries(r, "comm.read_rate").items() if 2 <= k <= R[-1]]) for r, _ in rs])
        ppar = mean([sum(d["posts"].values()) / max(1, sum(d["turns"].values())) for _, d in rs])
        sp = [r.spend for r, _ in rs]
        walls = [d["wall"] for _, d in rs]
        rows.append([arm, len(rs), f"{f(mean(acc))} ± {f(sd(acc))}", f(mean(con)), med, f(rr), f(ppar),
                     f"${mean([s['swarm'] for s in sp]):.3f} + ${mean([s['measurement'] for s in sp]):.3f}",
                     f(mean([s["calls"] for s in sp]), 0),
                     (f"{mean(walls) / 60:.1f} min" if mean(walls) is not None else NA)])
    L += ["## Summary", ""] + table(["arm", "n", "final acc (mean ± sd)", "final consensus", "rounds to 0.9 cons. (median)",
                                    f"read_rate r2-{R[-1]}", "posts/agent/round", "spend/run (swarm + meas.)", "calls/run",
                                    "wall/run"], rows) + [""]
    L += spend_lines([r for rs in arms.values() for r, _ in rs if r.dir.name not in simulated],
                     len(simulated))

    # trajectories: the belief and read-rate rows (when any run logs them), then every other
    # logged metric under its full name, then post_rate (from the events)
    logged = {n for rs in arms.values() for _, d in rs for n in d["metric_series"]}
    fixed = [(lab, n) for lab, n in TRAJECTORY_ROWS if n in logged]
    others = sorted(logged - {n for _, n in TRAJECTORY_ROWS})
    L += ["## Trajectories (mean over seeds)", ""]
    for arm, rs in sorted(arms.items()):
        rows = []
        for lab, n in [*fixed, *((n, n) for n in others)]:
            ss = [d["metric_series"].get(n, {}) for _, d in rs]
            rows.append([lab] + [f(mean([s.get(k) for s in ss]), 2) for k in R])
        rows.append(["post_rate"] + [f(mean([len(d["posters"][k]) / d["turns"][k]
                                              for _, d in rs if d["turns"][k]]), 2) for k in R])
        L += [f"**{arm}**", ""] + table(["metric"] + [f"r{k}" for k in R], rows) + [""]

    probe_ctx = {}
    if flag:  # FlagGame-style runs: committed `guess` actions and a verifiable truth
        # where the swarm went
        L += ["## Where the swarm went (final committed guesses)", ""]
        rows = []
        for arm, rs in sorted(arms.items()):
            for run, d in rs:
                try:
                    truth, rival, match = truth_info(run, d["agents"])
                except Exception as e:  # noqa: BLE001
                    truth = rival = match = None
                    L.append(f"(verify rebuild failed for {run.id}: {type(e).__name__}: {e})")
                probe_ctx[run.id] = (truth, rival)
                g = d["guess_by_round"].get(d["last_round"], {})
                n = max(len(d["agents"]), len(g)) or 1
                c = Counter(g.values())
                dist = ", ".join(f"{k}:{v}" for k, v in sorted(c.items(), key=lambda kv: -kv[1])) or NA
                if truth is not None:
                    dist = ", ".join(f"{k}{'(T)' if k == truth else '(R)' if k == rival else ''}:{v}"
                                     for k, v in sorted(c.items(), key=lambda kv: -kv[1]))
                sh = lambda k, c=c, n=n: f(c.get(k, 0) / n, 2) if k is not None else NA
                other = f((n - c.get(truth, 0) - c.get(rival, 0)) / n, 2) if truth is not None else NA
                rows.append([run.id, truth or NA, rival or NA, {True: "yes", False: "NO", None: NA}[match],
                             dist, sh(rival), sh(truth), other])
        L += table(["run", "truth", "rival", "matches score", "guess counts", "on rival", "on truth", "elsewhere"], rows) + [""]

        L += terminal_section(arms)  # M6

        # probes
        L += ["## Probe vs world belief", ""]
        for arm, rs in sorted(arms.items()):
            pc, pa, wc, wa, dis = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
            skips: Counter = Counter()
            for run, d in rs:
                try:
                    pr = run.probes.get("belief", [])
                except Exception:  # noqa: BLE001 - a broken run is reported as missing data
                    pr = []
                truth = probe_ctx.get(run.id, (None, None))[0]
                byr = defaultdict(dict)
                for r, ag, parsed, ok in pr:
                    if isinstance(parsed, dict) and "skipped" in parsed:  # not asked: not counted
                        skips[str(parsed["skipped"])] += 1
                        continue
                    byr[r][ag] = parsed.get("candidate") if ok and isinstance(parsed, dict) else None
                for r, ans in byr.items():
                    cnt = Counter(ans.values())
                    n = len(ans)
                    pc[r].append(max(cnt.values()) / n if n else None)
                    if truth is not None:
                        pa[r].append(sum(1 for v in ans.values() if v == truth) / n)
                    g = d["guess_by_round"].get(r, {})
                    both = [ag for ag in ans if ag in g]
                    if both:
                        dis[r].append(sum(1 for ag in both if ans[ag] != g[ag]) / len(both))
                for k, nm in ((wc, "belief.consensus"), (wa, "belief.accuracy")):
                    for r, v in mseries(run, nm).items():
                        k[r].append(v)
            rows = [[lab] + [f(mean(s.get(k, [])), 2) for k in R] for lab, s in
                    [("probe consensus", pc), ("world consensus", wc), ("probe accuracy", pa),
                     ("world accuracy", wa), ("disagreement (probe != guess)", dis)]]
            L += [f"**{arm}**", ""] + table(["metric"] + [f"r{k}" for k in R], rows) + [""]
            if skips:
                why = ", ".join(f"{k} {v}" for k, v in sorted(skips.items()))
                L += [(f"Probes skipped: {sum(skips.values())} over {len(rs)} run(s) ({why}); "
                       "skipped agents are left out of that round's probe columns."), ""]
            else:
                L += ["Probes skipped: none.", ""]

    # reading
    L += ["## Reading behaviour", ""]
    rows = []
    for arm, rs in sorted(arms.items()):
        share = {k: mean([d["reads"][k] / d["turns"][k] for _, d in rs if d["turns"][k]]) for k in R}
        per = mean([sum(d["read_deliv"].values()) / sum(d["reads"].values()) for _, d in rs if sum(d["reads"].values())])
        never = mean([1 - len(d["agents_read"]) / len(d["agents"]) for _, d in rs if d["agents"]])
        rows.append([arm] + [f(share[k], 2) for k in R] + [f(per, 1), f(never, 2)])
    L += table(["arm"] + [f"read-turn share r{k}" for k in R] + ["deliveries/read", "never-read agents"], rows) + [""]

    # health
    L += ["## Tool protocol health", ""]
    rows = []
    for arm, rs in sorted(arms.items()):
        y, fr = Counter(), Counter()
        for _, d in rs:
            y.update(d["yields"])
            fr.update(d["finish"])
        rows.append([arm, dict(y) or NA, dict(fr) or NA, sum(d["length"] for _, d in rs),
                     sum(d["err"] for _, d in rs),
                     f"{sum(d['cached'] for _, d in rs)}/{sum(d['nresp'] for _, d in rs)}",
                     sum(d["max_tokens"] for _, d in rs),
                     f"{sum(d['rejected_calls'] for _, d in rs)}/{sum(d['tool_returns'] for _, d in rs)}"])
    L += table(["arm", "yield kinds", "finish reasons (turn usage)", "length (responses)", "errored turns",
                "cache hits/responses", "max_tokens (responses)", "rejected tool calls"], rows) + [""]

    # protocol health, per arm, for any world
    L += ["## Protocol health", ""]
    for arm, rs in sorted(arms.items()):
        L += [f"**{arm}**", ""] + table(["measure", "value"], health_rows(rs)) + [""]

    if "coloring" in kinds:
        L += coloring_section(arms)
    return "\n".join(L)


def write_report(runs_dir: Path | str, out: Path | str | None = None,
                 title: str = DEFAULT_TITLE, include_fake: bool = False) -> str:
    """Build the report; also write it to `out` (with a trailing newline) when given."""
    text = build_report(runs_dir, title, include_fake=include_fake)
    if out is not None:
        Path(out).write_text(text + "\n")
    return text
