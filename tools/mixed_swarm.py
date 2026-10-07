"""One mixed swarm: 8 Haiku + 8 Qwen agents on one board (M3a real test). SPENDS MONEY.

    SWARMLAB_REAL=1 uv run python tools/mixed_swarm.py [--out runs/m3-tests] [--seed 1] [--timeout 90]

FlagGame N=16, 6 rounds, broadcast board, a BeliefProbe every round. Agents a000-a007 are
`LLMAgent(model="anthropic:claude-haiku-4-5", max_tokens=1024)`, a008-a015 are
`LLMAgent(model="hf:Qwen/Qwen3.5-9B:deepinfra", max_tokens=2048,
extra={"chat_template_kwargs": {"enable_thinking": False}})` on the HF router (pricing pinned to
the router's DeepInfra listing, as in tools/real_smoke.py). `Budget(soft_usd=1.5, hard_usd=2.0,
measurement_usd=0.5)`. `--timeout` sets the hf provider's `timeout_s` (default: the provider
default, 90 s).

Report per model group: read rate (turns with a read_board call / turns), posts, final accuracy
(held world guesses), probe parse rate, latency median/p90 (non-cached responses), responses with
attempts > 1, provider errors, and cross-model reads (a read returning a post by the other group).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

from swarmlab import Board, Budget, Experiment
from swarmlab.metrics.belief import Accuracy, Consensus
from swarmlab.participants import LLMAgent
from swarmlab.probes import BeliefProbe
from swarmlab.providers.base import DEFAULT_TIMEOUT_S
from swarmlab.providers.openai_compat import OpenAICompatProvider
from swarmlab.world.flaggame import FlagGame

HAIKU = "anthropic:claude-haiku-4-5"
QWEN_ID = "Qwen/Qwen3.5-9B:deepinfra"
QWEN_PRICING = (0.10, 0.15, 0.10)
ROUNDS = 6
BUDGET = Budget(soft_usd=1.5, hard_usd=2.0, measurement_usd=0.5)
GROUPS = {"haiku": [f"a{i:03d}" for i in range(8)], "qwen": [f"a{i:03d}" for i in range(8, 16)]}


def experiment(timeout_s: float) -> Experiment:
    kw = {} if timeout_s == DEFAULT_TIMEOUT_S else {"timeout_s": timeout_s}
    return Experiment(
        name="mixed-swarm", world=FlagGame(),
        participants=[LLMAgent(model=HAIKU, max_tokens=1024)] * 8
        + [LLMAgent(model=f"hf:{QWEN_ID}", max_tokens=2048,
                    extra={"chat_template_kwargs": {"enable_thinking": False}})] * 8,
        medium=Board(topology="broadcast"),
        metrics=["belief.accuracy", "belief.consensus", "comm.read_rate", "comm.posts_per_round",
                 Accuracy(source="probe:belief"), Consensus(source="probe:belief")],
        probes=[BeliefProbe()], budget=BUDGET,
        providers={"hf": OpenAICompatProvider("hf", pricing={QWEN_ID: QWEN_PRICING}, **kw)},
    )


def pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 2)


def report(run) -> dict:
    group_of = {a: g for g, agents in GROUPS.items() for a in agents}
    ev = list(run.events)
    truth = run.score["truth"]
    held: dict[str, str] = {}
    for e in ev:
        if e["type"] == "action_committed" and e["action"]["name"] == "guess" and e["accepted"]:
            held[e["agent"]] = e["action"]["args"]["candidate"]
    author = {e["post_id"]: e["agent"] for e in ev if e["type"] == "post"}
    dpost = {e["delivery_id"]: e["post_id"] for e in ev if e["type"] == "delivery"}
    read_turns = {(e["round"], e["agent"]) for e in ev
                  if e["type"] == "tool_called" and e["tool"] == "read_board"}
    cross: Counter = Counter()
    for e in ev:
        if e["type"] == "read":
            for d in e["delivery_ids"]:
                a = author.get(dpost.get(d, ""))
                if a is not None:
                    cross[(group_of[e["agent"]], group_of[a])] += 1
    responses = [e for e in run.events_all if e.type == "inference_response"]
    out: dict = {"run": run.id, "status": run.status, "end": run.end_reason, "spend": run.spend,
                 "score": run.score, "groups": {},
                 "reads_by_reader_author_group": {f"{r}<-{a}": n for (r, a), n in sorted(cross.items())},
                 "metrics": {m: [round(v, 3) if v is not None else None for _, v, _ in vals]
                             for m, vals in run.metrics.items()}}
    probes = run.probes.get("belief", [])
    for g, agents in GROUPS.items():
        turns = [e for e in ev if e["type"] == "turn_ended" and e["agent"] in agents]
        rs = [e for e in responses if e.agent in agents]
        lat = [e.latency_s for e in rs if not e.cached and not e.finish_reason.startswith("error")]
        gp = [p for p in probes if p[1] in agents]
        out["groups"][g] = {
            "turns": len(turns),
            "read_rate": round(sum(1 for e in turns if (e["round"], e["agent"]) in read_turns)
                               / max(1, len(turns)), 3),
            "posts": sum(1 for e in ev if e["type"] == "post" and e["agent"] in agents),
            "accuracy": round(sum(1 for a in agents if held.get(a) == truth) / len(agents), 3),
            "guessed": sum(1 for a in agents if a in held),
            "probe_parse": f"{sum(1 for p in gp if p[3])}/{len(gp)}",
            "latency_median_s": round(statistics.median(lat), 2) if lat else None,
            "latency_p90_s": pct(lat, 0.9), "latency_max_s": round(max(lat), 2) if lat else None,
            "responses": len(rs),
            "retried": sum(1 for e in rs if e.attempts > 1),
            "attempts_hist": dict(Counter(e.attempts for e in rs)),
            "errors": dict(Counter(e.finish_reason for e in rs if e.finish_reason.startswith("error"))),
            "finish": dict(Counter(e.finish_reason for e in rs)),
            "yield_kinds": dict(Counter(e["yield_kind"] for e in turns)),
            "cost_usd": round(sum(e.cost_usd for e in rs), 4),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("runs/m3-tests"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    args = ap.parse_args()
    exp = experiment(args.timeout)
    est = exp.estimate(args.seed, ROUNDS)
    print(f"estimate ${est['usd']:.3f} by model {est['by_model']}; hard ceiling ${BUDGET.hard_usd}")
    if os.environ.get("SWARMLAB_REAL") != "1":
        print("SWARMLAB_REAL is not 1: not calling any provider.")
        return 0
    run = exp.run(args.seed, ROUNDS, out=args.out)
    rep = report(run)
    (run.dir / "report.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
