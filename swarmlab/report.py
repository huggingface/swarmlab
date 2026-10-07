"""`swarmlab report RUNS_DIR [--out FILE] [--title T]`: a Markdown report over finished runs.

Generalised from `tools/m2_report.py` (the M2 phase-1 Flag Game report); on the archived M2 runs
with `--title "M2 phase-1 Flag Game report"` it reproduces
`docs/notes/m2-phase1-report-2026-10-06.md` byte for byte. Read-only over the runs dir.

Decisions:

- Every run directory directly under RUNS_DIR whose `run.json` says `status: ended` is loaded with
  `Run.load` (a full replay: a run that does not replay is listed as skipped with the error).
  Runs are grouped by arm (the middle part of `<experiment>__<arm>__s<seed>`, else `run.json`'s
  arm, else `?`).
- Per-round columns span rounds 1..max(last committed round) over the runs (10 for M2), and the
  summary's read rate averages rounds 2..that maximum.
- Sections: summary, trajectories (belief and read-rate metrics; `n/a` where a metric is not
  logged), reading behaviour and tool-protocol health for every world; "where the swarm went"
  and "probe vs world belief" only when some run has committed `guess` actions (FlagGame-style
  worlds; truth and rival come from rebuilding the world, since `verify()` is not persisted).
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
         "t1": None, "turn_n": Counter(), "length": 0}
    cur = {}
    last_round = 0
    for ev in run.events_all:
        t, r, a = ev.type, ev.round, ev.agent
        if t == "run_started":
            d["t0"] = ev.ts
        elif t == "run_ended":
            d["t1"] = ev.ts
        elif t == "turn_ended":
            d["turns"][r] += 1
            d["agents"].add(a)
            d["yields"][ev.yield_kind] += 1
            if ev.yield_kind == "error" or getattr(ev, "error", None):
                d["err"] += 1
            for fr in (ev.usage or {}).get("finish_reasons", []) if isinstance(ev.usage, dict) else []:
                d["finish"][fr] += 1
        elif t == "read":
            d["reads"][r] += 1
            d["read_deliv"][r] += len(ev.delivery_ids)
            d["agents_read"].add(a)
        elif t == "post":
            d["posts"][r] += 1
        elif t == "inference_response":
            d["nresp"] += 1
            d["cached"] += bool(getattr(ev, "cached", False))
            fr = getattr(ev, "finish_reason", None)
            if fr == "length":
                d["length"] += 1
        elif t == "action_committed":
            act = ev.action if isinstance(ev.action, dict) else dict(ev.action)
            if act.get("name") == "guess" and ev.accepted:
                cur[str(a)] = act["args"]["candidate"]
        elif t == "round_committed":
            d["guess_by_round"][r] = dict(cur)
            last_round = r
    d["last_round"] = last_round
    d["wall"] = (d["t1"] - d["t0"]) if d["t0"] and d["t1"] else None
    return d


def truth_info(run, agents):
    """verify() is not persisted: rebuild the world exactly as the runner does."""
    seed = run.spec.options.seed
    exp = Experiment.from_spec(run.spec)
    ag = [p.agent for p in exp.participants if getattr(p, "agent", None)] or sorted(agents)
    exp.world.reset(derive(seed, "world"), ag)
    v = exp.world.verify()
    rec = run.score.get("truth")
    return v["truth"], v["rival"], v["truth"] == rec if rec is not None else None


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
    flag = any(d["guess_by_round"].get(d["last_round"]) for rs in arms.values() for _, d in rs)
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

    # trajectories
    L += ["## Trajectories (mean over seeds)", ""]
    for arm, rs in sorted(arms.items()):
        rows = []
        for lab, n in [("accuracy", "belief.accuracy"), ("consensus", "belief.consensus"),
                       ("entropy", "belief.entropy"), ("read_rate", "comm.read_rate")]:
            ss = [mseries(r, n) for r, _ in rs]
            rows.append([lab] + [f(mean([s.get(k) for s in ss]), 2) for k in R])
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

        # probes
        L += ["## Probe vs world belief", ""]
        for arm, rs in sorted(arms.items()):
            pc, pa, wc, wa, dis = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
            for run, d in rs:
                try:
                    pr = run.probes.get("belief", [])
                except Exception:  # noqa: BLE001 - a broken run is reported as missing data
                    pr = []
                truth = probe_ctx.get(run.id, (None, None))[0]
                byr = defaultdict(dict)
                for r, ag, parsed, ok in pr:
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
                     f"{sum(d['cached'] for _, d in rs)}/{sum(d['nresp'] for _, d in rs)}"])
    L += table(["arm", "yield kinds", "finish reasons (turn usage)", "length (responses)", "errored turns",
                "cache hits/responses"], rows) + [""]
    return "\n".join(L)


def write_report(runs_dir: Path | str, out: Path | str | None = None,
                 title: str = DEFAULT_TITLE, include_fake: bool = False) -> str:
    """Build the report; also write it to `out` (with a trailing newline) when given."""
    text = build_report(runs_dir, title, include_fake=include_fake)
    if out is not None:
        Path(out).write_text(text + "\n")
    return text
