"""Paired Haiku run from round 0 (M3a §2 real test). SPENDS MONEY; operator approval required.

    SWARMLAB_REAL=1 uv run python tools/paired_haiku.py [--out runs/m3-tests]

FlagGame N=8, 6 rounds, broadcast board, `LLMAgent(model="anthropic:claude-haiku-4-5",
max_tokens=1024)`, a BeliefProbe every round. The patched experiment moves agent a003's crop
(`FlagGame(crop_overrides=...)`) to a window whose content is contained in the RIVAL as well as
the truth (so it no longer tells the twins apart). The window is computed from the world rebuilt
from the spec: among windows whose containing set includes the rival, prefer the containing set
{truth, rival} exactly, then the smallest set, then scan order. If seed 1 has no such window (or
a003's own crop is already that ambiguous) seeds 2..5 are tried. One `Experiment.pair` with
repeats=1 and control=True: base, patched, control. `Budget(soft_usd=1.2, hard_usd=1.5,
measurement_usd=0.4)` per run, so the worst case is 3 x 1.5 = $4.50.

Without SWARMLAB_REAL=1 it prints the chosen seed, window and estimates, and exits.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from swarmlab import Board, Budget, Experiment
from swarmlab.metrics.belief import Accuracy, Consensus
from swarmlab.participants import LLMAgent
from swarmlab.probes import BeliefProbe
from swarmlab.rng import derive
from swarmlab.world.flaggame import FlagGame, candidates_containing

HAIKU = "anthropic:claude-haiku-4-5"
N, ROUNDS, TARGET = 8, 6, "a003"
BUDGET = Budget(soft_usd=1.2, hard_usd=1.5, measurement_usd=0.4)
TOTAL_CAP_USD = 5.0
METRICS = ["belief.accuracy", "belief.consensus",
           Accuracy(source="probe:belief"), Consensus(source="probe:belief")]
AGENTS = [f"a{i:03d}" for i in range(N)]


def experiment(world: FlagGame, arm: str) -> Experiment:
    return Experiment(
        name="paired-haiku", arm=arm, world=world,
        participants=[LLMAgent(model=HAIKU, max_tokens=1024)] * N,
        medium=Board(topology="broadcast"), metrics=list(METRICS),
        probes=[BeliefProbe()], budget=BUDGET,
    )


def window_at(world: FlagGame, y: int, x: int) -> list[str]:
    truth = world.candidates[world.truth]
    return [r[x:x + world.crop_w] for r in truth[y:y + world.crop_h]]


def choose(seed: int, exp: Experiment) -> dict | None:
    """Rebuild the world from the spec, reset it as the runner does, and pick the window."""
    world = Experiment.from_spec(exp.to_spec(seed=seed, max_rounds=ROUNDS)).world
    world.reset(derive(seed, "world"), AGENTS)
    own = candidates_containing(world.candidates, world.crop_rows(TARGET))
    best = None
    for y in range(world.height - world.crop_h + 1):
        for x in range(world.width - world.crop_w + 1):
            cont = candidates_containing(world.candidates, window_at(world, y, x))
            if world.rival not in cont or len(cont) < 2:
                continue
            key = (set(cont) != {world.truth, world.rival}, len(cont))
            if best is None or key < best[0]:
                best = (key, [y, x], cont)
    if best is None or world.rival in own:  # nothing to move to, or a003 is already ambiguous
        return None
    return {"seed": seed, "truth": world.truth, "rival": world.rival, "window": best[1],
            "window_contained_in": best[2], "window_rows": window_at(world, *best[1]),
            "own_crop": list(world.crops[TARGET]), "own_contained_in": own,
            "own_rows": world.crop_rows(TARGET),
            "crops_unique_to_truth": sum(
                1 for a in AGENTS
                if candidates_containing(world.candidates, world.crop_rows(a)) == [world.truth])}


# ---- analysis ------------------------------------------------------------------------------------
def guesses_by_round(run, agent: str | None = None) -> dict[str, dict[int, str]]:
    out: dict[str, dict[int, str]] = {}
    for e in run.events:
        if (e["type"] == "action_committed" and e["action"]["name"] == "guess" and e["accepted"]
                and (agent is None or e["agent"] == agent)):
            out.setdefault(e["agent"], {})[e["round"]] = e["action"]["args"]["candidate"]
    return out


def held(guesses: dict[int, str], rounds: int) -> list[str | None]:
    """The latest guess held at the end of each round 1..rounds."""
    cur, out = None, []
    for r in range(1, rounds + 1):
        cur = guesses.get(r, cur)
        out.append(cur)
    return out


def probe_beliefs(run, agent: str) -> list[str | None]:
    return [p.get("candidate") for r, a, p, ok in run.probes.get("belief", []) if a == agent]


def reads_of(run, author: str) -> list[tuple[int, str]]:
    """(round, reader) for every read that returned a post by `author`."""
    posts = {e["post_id"] for e in run.events if e["type"] == "post" and e["agent"] == author}
    dpost = {e["delivery_id"]: e["post_id"] for e in run.events if e["type"] == "delivery"}
    out = []
    for e in run.events:
        if e["type"] == "read" and any(dpost.get(d) in posts for d in e["delivery_ids"]):
            out.append((e["round"], e["agent"]))
    return out


def series(run, name: str) -> list:
    return [round(v, 3) if v is not None else None for _, v, _ in run.metrics.get(name, [])]


def summary(run) -> dict:
    turns = [e for e in run.events if e["type"] == "turn_ended"]
    return {"run": run.id, "status": run.status, "end": run.end_reason, "spend": run.spend,
            "score": run.score, "yield_kinds": dict(Counter(e["yield_kind"] for e in turns)),
            "posts_by_target": [e["text"] for e in run.events
                                if e["type"] == "post" and e["agent"] == TARGET]}


def report(res, choice: dict) -> dict:
    roles = {"base": res.base[0], "patched": res.patched[0], "control": res.controls[0]}
    out: dict = {"choice": choice, "runs": {}}
    for role, run in roles.items():
        g = guesses_by_round(run)
        out["runs"][role] = {
            **summary(run),
            "accuracy": series(run, "belief.accuracy"),
            "consensus": series(run, "belief.consensus"),
            "probe_accuracy": series(run, "belief.accuracy@probe:belief"),
            "probe_consensus": series(run, "belief.consensus@probe:belief"),
            "target_guesses": held(g.get(TARGET, {}), ROUNDS),
            "target_probe": probe_beliefs(run, TARGET),
            "held_guesses": {a: held(g.get(a, {}), ROUNDS) for a in AGENTS},
            "reads_of_target_posts": reads_of(run, TARGET),
        }
    base = out["runs"]["base"]["held_guesses"]
    for role in ("patched", "control"):
        other = out["runs"][role]["held_guesses"]
        out["runs"][role]["others_differing_from_base"] = sorted(
            a for a in AGENTS if a != TARGET and other[a] != base[a])
    out["effect"] = {m: res.effect(m) for m in
                     ("belief.consensus", "belief.accuracy", "belief.consensus@probe:belief",
                      "belief.accuracy@probe:belief")}
    out["total_spend"] = sum(r["spend"]["swarm"] + r["spend"]["measurement"]
                             for r in out["runs"].values())
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("runs/m3-tests"))
    args = ap.parse_args()
    choice = None
    for seed in range(1, 6):
        choice = choose(seed, experiment(FlagGame(), "base"))
        if choice is not None:
            break
        print(f"seed {seed}: no rival-contained window for {TARGET} (or its crop is already ambiguous)")
    if choice is None:
        print("no seed in 1..5 works")
        return 1
    print(json.dumps(choice, indent=1))
    base = experiment(FlagGame(), "base")
    patched = experiment(FlagGame(crop_overrides={TARGET: choice["window"]}), "patched")
    est = base.estimate(choice["seed"], ROUNDS)
    print(f"estimate per run ${est['usd']:.3f} (2 calls/turn x 3000 prompt tokens); "
          f"hard ceiling ${BUDGET.hard_usd} x 3 runs = ${3 * BUDGET.hard_usd:.2f} worst case")
    if 3 * BUDGET.hard_usd >= TOTAL_CAP_USD:
        return 1
    if os.environ.get("SWARMLAB_REAL") != "1":
        print("SWARMLAB_REAL is not 1: not calling any provider.")
        return 0
    res = base.pair(choice["seed"], ROUNDS, patch=patched, repeats=1, control=True, out=args.out)
    rep = report(res, choice)
    (res.dir / "report.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(json.dumps(rep, indent=1, default=str))
    print(f"\npair dir {res.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
