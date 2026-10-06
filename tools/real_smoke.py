"""Real-provider smoke test (docs/INTERFACE-M1b.md §9 item 7). SPENDS MONEY; operator approval required.

Runs only when SWARMLAB_REAL=1; otherwise it prints the worst-case estimate and exits.

    SWARMLAB_REAL=1 uv run python tools/real_smoke.py [--out runs/real_smoke] [--seed 1]

Two arms, each N=4 LLMAgents on the Flag Game, 3 rounds, broadcast board, a BeliefProbe every
round, `Budget(soft_usd=0.40, hard_usd=0.50, measurement_usd=0.10)` per arm:

- `haiku`: `anthropic:claude-haiku-4-5` (ANTHROPIC_API_KEY or ANTHROPIC_KEY), preset pricing
  (1.00, 5.00, 0.10) per M tokens.
- `qwen`: `hf:Qwen/Qwen3.5-9B:deepinfra` on the HF router (HF_TOKEN), pinned to DeepInfra with the
  router's `model:provider` suffix. Chosen on 2026-10-06 from `GET router.huggingface.co/v1/models`:
  the smallest Qwen3.x >= 7B whose listing advertises `supports_tools` on DeepInfra (Qwen/Qwen3-8B
  is 1B smaller but is tool-listed only on nscale). Pricing is the router's listed DeepInfra price,
  input 0.10 / output 0.15 USD per M tokens; no cached-input price is listed, so cached prompt
  tokens are priced at the input price: (0.10, 0.15, 0.10).

Per arm it prints: the estimate, spend (swarm, measurement, calls), cost per turn, tool-call
success rate, yield kinds, probe ok rate, accuracy by round (world and probe), and whether the
native tool protocol worked (at least one native tool call was executed with ok=True and at least
one `guess` was accepted, and no response had unparseable tool arguments).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

from swarmlab import Board, Budget, Experiment
from swarmlab.metrics.belief import Accuracy, Consensus
from swarmlab.participants import LLMAgent
from swarmlab.probes import BeliefProbe
from swarmlab.providers.openai_compat import OpenAICompatProvider
from swarmlab.worlds import FlagGame

N, ROUNDS = 4, 3
BUDGET = Budget(soft_usd=0.40, hard_usd=0.50, measurement_usd=0.10)
QWEN_ID = "Qwen/Qwen3.5-9B:deepinfra"
QWEN_PRICING = (0.10, 0.15, 0.10)  # router-listed DeepInfra price (input, output, cached=input)
ARMS = {
    "haiku": "anthropic:claude-haiku-4-5",
    "qwen": f"hf:{QWEN_ID}",
}
AGENT = {"max_tokens": 1024, "max_calls": 6}


def experiment(arm: str) -> Experiment:
    return Experiment(
        name="real-smoke",
        arm=arm,
        world=FlagGame(n_candidates=8),
        participants=[LLMAgent(model=ARMS[arm], **AGENT)] * N,
        medium=Board(topology="broadcast"),
        metrics=["belief.accuracy", "belief.consensus",
                 Accuracy(source="probe:belief"), Consensus(source="probe:belief")],
        probes=[BeliefProbe()],
        budget=BUDGET,
        providers={"hf": OpenAICompatProvider("hf", pricing={QWEN_ID: QWEN_PRICING})},
    )


def estimate(exp: Experiment) -> dict:
    return exp.estimate(seed=1, max_rounds=ROUNDS, calls_per_turn=AGENT["max_calls"])


def report(arm: str, run) -> None:
    events = list(run.events)
    called = [e for e in events if e["type"] == "tool_called"]
    returned = [e for e in events if e["type"] == "tool_returned"]
    ok_calls = sum(1 for e in returned if (e["result"] or {}).get("ok"))
    turns = [e for e in events if e["type"] == "turn_ended"]
    kinds = Counter(e["yield_kind"] for e in turns)
    probes = run.probes.get("belief", [])
    probe_ok = sum(1 for _, _, _, ok in probes if ok)
    guesses = [e for e in events if e["type"] == "action_committed" and e["action"]["name"] == "guess"]
    accepted = sum(1 for e in guesses if e["accepted"])
    responses = [e for e in run.events_all if e.type == "inference_response"]
    bad_args = sum(1 for e in responses if e.finish_reason == "bad_tool_args")
    errors = [e for e in responses if e.finish_reason.startswith("error")]
    turn_cost = [((e["usage"] or {}).get("cost_usd") or 0.0) for e in turns]
    native = bool(called) and ok_calls > 0 and accepted > 0 and bad_args == 0
    print(f"\n== {arm}: {ARMS[arm]}  status={run.status} end={run.end_reason}")
    print(f"  spend          {run.spend}")
    print(f"  cost per turn  mean ${sum(turn_cost) / max(1, len(turn_cost)):.5f} "
          f"max ${max(turn_cost, default=0.0):.5f} over {len(turns)} turns")
    print(f"  tool calls     {len(called)}, ok {ok_calls} "
          f"({ok_calls / max(1, len(called)):.0%}); by tool {dict(Counter(e['tool'] for e in called))}")
    print(f"  yield kinds    {dict(kinds)}")
    print(f"  guesses        {len(guesses)}, accepted {accepted}")
    print(f"  probes         {len(probes)}, ok {probe_ok} ({probe_ok / max(1, len(probes)):.0%})")
    print(f"  responses      {len(responses)}, bad_tool_args {bad_args}, errors {len(errors)}, "
          f"served_by {dict(Counter(e.served_by for e in responses))}")
    for name in ("belief.accuracy", "belief.accuracy@probe:belief"):
        print(f"  {name:28s} {[(r, v) for r, v, _ in run.metrics.get(name, [])]}")
    print(f"  native tool protocol worked: {native}")
    errs = [e for e in turns if e.get("error")]
    if errs:
        print(f"  first turn error: {errs[0]['error'][-400:]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("runs/real_smoke"))
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    exps = {arm: experiment(arm) for arm in ARMS}
    total = 0.0
    for arm, exp in exps.items():
        est = estimate(exp)
        total += est["usd"]
        print(f"estimate {arm}: worst-case ${est['usd']:.4f} ({est['calls']} turn calls + "
              f"{est['probe_calls']} probe calls); hard ceiling ${BUDGET.hard_usd:.2f}")
    print(f"estimate total: ${total:.4f}; absolute cap {len(exps)} x ${BUDGET.hard_usd:.2f}")
    if os.environ.get("SWARMLAB_REAL") != "1":
        print("SWARMLAB_REAL is not 1: not calling any provider.")
        return 0
    out = args.out / time.strftime("%Y%m%d-%H%M%S")
    for arm, exp in exps.items():
        try:
            run = exp.run(seed=args.seed, max_rounds=ROUNDS, out=out)
        except Exception as e:  # noqa: BLE001 - report and continue with the other arm
            print(f"\n== {arm}: FAILED {type(e).__name__}: {e}")
            continue
        report(arm, run)
    print(f"\nrun dirs under {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
