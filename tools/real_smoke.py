"""Real-provider smoke test (docs/INTERFACE-M1b.md §9 item 7). SPENDS MONEY; operator approval required.

Runs only when SWARMLAB_REAL=1; otherwise it prints the worst-case estimate and exits.

    SWARMLAB_REAL=1 uv run python tools/real_smoke.py [--out runs/real_smoke] [--seed 1]

Three arms of LLMAgents on the Flag Game, broadcast board, a BeliefProbe every round, every agent
at `max_tokens=1024, max_calls=6`. `haiku` and `qwen` keep `Budget(soft_usd=0.40, hard_usd=0.50,
measurement_usd=0.10)`; the small `haiku-json` arm gets `Budget(soft_usd=0.15, hard_usd=0.20,
measurement_usd=0.05)`. The script refuses to run if any arm's estimate exceeds its hard ceiling
or the summed worst-case estimate reaches $1.00.

- `haiku`: N=4, 3 rounds, `anthropic:claude-haiku-4-5` (ANTHROPIC_API_KEY or ANTHROPIC_KEY),
  preset pricing (1.00, 5.00, 0.10) per M tokens.
- `qwen`: `hf:Qwen/Qwen3.5-9B:deepinfra` on the HF router (HF_TOKEN), pinned to DeepInfra with the
  router's `model:provider` suffix. Chosen on 2026-10-06 from `GET router.huggingface.co/v1/models`:
  the smallest Qwen3.x >= 7B whose listing advertises `supports_tools` on DeepInfra (Qwen/Qwen3-8B
  is 1B smaller but is tool-listed only on nscale). Pricing is the router's listed DeepInfra price,
  input 0.10 / output 0.15 USD per M tokens; no cached-input price is listed, so cached prompt
  tokens are priced at the input price: (0.10, 0.15, 0.10). N=4, 3 rounds. Sent with
  `extra={"chat_template_kwargs": {"enable_thinking": False}}` (DeepInfra's documented body field
  for Qwen3.5-9B): in the first smoke (2026-10-06) thinking used the whole 1024-token budget on 7
  of 12 turns and 8 of 12 probes (`finish_reason="length"`, empty text).
- `haiku-json`: N=2, 2 rounds, Haiku with `tool_protocol="json"` (no native tools sent), to learn
  whether the JSON protocol works on Haiku too.

Per arm it prints: the estimate, spend (swarm, measurement, calls), cost per turn, tool-call
success rate, yield kinds, turn notes (`length`, `text_tool_fallback`), finish reasons, probe ok
rate, accuracy by round (world and probe), and whether the arm's tool protocol worked (at least
one tool call was executed with ok=True and at least one `guess` was accepted, and no response
had unparseable tool arguments).
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

BUDGET = Budget(soft_usd=0.40, hard_usd=0.50, measurement_usd=0.10)
SMALL_BUDGET = Budget(soft_usd=0.15, hard_usd=0.20, measurement_usd=0.05)
TOTAL_CAP_USD = 1.00  # worst case over all arms must stay under this
QWEN_ID = "Qwen/Qwen3.5-9B:deepinfra"
QWEN_PRICING = (0.10, 0.15, 0.10)  # router-listed DeepInfra price (input, output, cached=input)
HAIKU = "anthropic:claude-haiku-4-5"
AGENT = {"max_tokens": 1024, "max_calls": 6}
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}
ARMS: dict[str, dict] = {
    "haiku": {"model": HAIKU, "n": 4, "rounds": 3, "agent": {}},
    "qwen": {"model": f"hf:{QWEN_ID}", "n": 4, "rounds": 3, "agent": {"extra": NO_THINKING}},
    "haiku-json": {"model": HAIKU, "n": 2, "rounds": 2, "agent": {"tool_protocol": "json"},
                   "budget": SMALL_BUDGET},
}


def experiment(arm: str) -> Experiment:
    cfg = ARMS[arm]
    return Experiment(
        name="real-smoke",
        arm=arm,
        world=FlagGame(n_candidates=8),
        participants=[LLMAgent(model=cfg["model"], **AGENT, **cfg["agent"])] * cfg["n"],
        medium=Board(topology="broadcast"),
        metrics=["belief.accuracy", "belief.consensus",
                 Accuracy(source="probe:belief"), Consensus(source="probe:belief")],
        probes=[BeliefProbe()],
        budget=cfg.get("budget", BUDGET),
        providers={"hf": OpenAICompatProvider("hf", pricing={QWEN_ID: QWEN_PRICING})},
    )


def estimate(arm: str, exp: Experiment) -> dict:
    return exp.estimate(seed=1, max_rounds=ARMS[arm]["rounds"], calls_per_turn=AGENT["max_calls"])


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
    worked = bool(called) and ok_calls > 0 and accepted > 0 and bad_args == 0
    notes = Counter(n.split(":")[0] for e in turns for n in (e["usage"] or {}).get("notes", []))
    finishes = Counter(e.finish_reason for e in responses)
    cfg = ARMS[arm]
    print(f"\n== {arm}: {cfg['model']} {cfg['agent']}  status={run.status} end={run.end_reason}")
    print(f"  spend          {run.spend}")
    print(f"  cost per turn  mean ${sum(turn_cost) / max(1, len(turn_cost)):.5f} "
          f"max ${max(turn_cost, default=0.0):.5f} over {len(turns)} turns")
    print(f"  tool calls     {len(called)}, ok {ok_calls} "
          f"({ok_calls / max(1, len(called)):.0%}); by tool {dict(Counter(e['tool'] for e in called))}")
    print(f"  yield kinds    {dict(kinds)}")
    print(f"  turn notes     {dict(notes)}; finish reasons {dict(finishes)}")
    print(f"  guesses        {len(guesses)}, accepted {accepted}")
    print(f"  probes         {len(probes)}, ok {probe_ok} ({probe_ok / max(1, len(probes)):.0%})")
    print(f"  responses      {len(responses)}, bad_tool_args {bad_args}, errors {len(errors)}, "
          f"served_by {dict(Counter(e.served_by for e in responses))}")
    for name in ("belief.accuracy", "belief.accuracy@probe:belief"):
        print(f"  {name:28s} {[(r, v) for r, v, _ in run.metrics.get(name, [])]}")
    protocol = cfg["agent"].get("tool_protocol", "native")
    print(f"  {protocol} tool protocol worked: {worked}")
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
        est = estimate(arm, exp)
        total += est["usd"]
        print(f"estimate {arm}: worst-case ${est['usd']:.4f} ({est['calls']} turn calls + "
              f"{est['probe_calls']} probe calls); hard ceiling "
              f"${ARMS[arm].get('budget', BUDGET).hard_usd:.2f}")
        hard = ARMS[arm].get("budget", BUDGET).hard_usd
        if est["usd"] > hard:
            print(f"estimate for {arm} exceeds its hard ceiling: not running.")
            return 1
    caps = sum(ARMS[a].get("budget", BUDGET).hard_usd for a in exps)
    print(f"estimate total: ${total:.4f}; must stay under ${TOTAL_CAP_USD:.2f} "
          f"(sum of hard ceilings ${caps:.2f})")
    if total >= TOTAL_CAP_USD:
        print("total worst case is over the cap: not running.")
        return 1
    if os.environ.get("SWARMLAB_REAL") != "1":
        print("SWARMLAB_REAL is not 1: not calling any provider.")
        return 0
    out = args.out / time.strftime("%Y%m%d-%H%M%S")
    for arm, exp in exps.items():
        try:
            run = exp.run(seed=args.seed, max_rounds=ARMS[arm]["rounds"], out=out)
        except Exception as e:  # noqa: BLE001 - report and continue with the other arm
            print(f"\n== {arm}: FAILED {type(e).__name__}: {e}")
            continue
        report(arm, run)
    print(f"\nrun dirs under {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
