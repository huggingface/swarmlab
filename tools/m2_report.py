#!/usr/bin/env python
"""M2 phase-1 Flag Game report. Read-only over the runs dir. Usage: m2_report.py RUNS_DIR [--out FILE]"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from swarmlab.experiment import Experiment, Run
from swarmlab.rng import derive

NA = "n/a"
R = range(1, 11)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("--out")
    a = ap.parse_args()
    arms = defaultdict(list)
    skipped = []
    for d in sorted(Path(a.runs_dir).iterdir()):
        rj = d / "run.json"
        if not rj.is_file():
            continue
        try:
            if json.loads(rj.read_text()).get("status") != "ended":
                skipped.append(d.name)
                continue
            run = Run.load(d)
        except Exception as e:  # noqa: BLE001
            skipped.append(f"{d.name} ({type(e).__name__})")
            continue
        parts = run.id.split("__")
        arm = parts[1] if len(parts) >= 3 else (run.meta.get("arm") or "?")
        arms[arm].append((run, scan(run)))
    L = ["# M2 phase-1 Flag Game report", "", f"Runs dir: `{a.runs_dir}`; skipped (not ended / unreadable): {', '.join(skipped) or 'none'}", ""]

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
        rr = mean([mean([v for k, v in mseries(r, "comm.read_rate").items() if 2 <= k <= 10]) for r, _ in rs])
        ppar = mean([sum(d["posts"].values()) / max(1, sum(d["turns"].values())) for _, d in rs])
        sp = [r.spend for r, _ in rs]
        walls = [d["wall"] for _, d in rs]
        rows.append([arm, len(rs), f"{f(mean(acc))} ± {f(sd(acc))}", f(mean(con)), med, f(rr), f(ppar),
                     f"${mean([s['swarm'] for s in sp]):.3f} + ${mean([s['measurement'] for s in sp]):.3f}",
                     f(mean([s["calls"] for s in sp]), 0),
                     (f"{mean(walls) / 60:.1f} min" if mean(walls) is not None else NA)])
    L += ["## Summary", ""] + table(["arm", "n", "final acc (mean ± sd)", "final consensus", "rounds to 0.9 cons. (median)",
                                    "read_rate r2-10", "posts/agent/round", "spend/run (swarm + meas.)", "calls/run",
                                    "wall/run"], rows) + [""]

    # trajectories
    L += ["## Trajectories (mean over seeds)", ""]
    for arm, rs in sorted(arms.items()):
        rows = []
        for lab, n in [("accuracy", "belief.accuracy"), ("consensus", "belief.consensus"),
                       ("entropy", "belief.entropy"), ("read_rate", "comm.read_rate")]:
            ss = [mseries(r, n) for r, _ in rs]
            rows.append([lab] + [f(mean([s.get(k) for s in ss]), 2) for k in R])
        L += [f"**{arm}**", ""] + table(["metric"] + [f"r{k}" for k in R], rows) + [""]

    # where the swarm went
    L += ["## Where the swarm went (final committed guesses)", ""]
    rows, probe_ctx = [], {}
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
    text = "\n".join(L)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
