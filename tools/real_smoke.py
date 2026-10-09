"""Real-provider smoke test (docs/INTERFACE-M1b.md §9 item 7). SPENDS MONEY; operator approval required.

Runs only when SWARMLAB_REAL=1; otherwise it prints the worst-case estimate and exits.

    SWARMLAB_REAL=1 uv run python tools/real_smoke.py [--out runs/real_smoke] [--seed 1]
        [--arms haiku,qwen,haiku-json | --arms haiku-image | --arms gemma-image,gemma-manager]

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
- `haiku-image` (M5 §5): N=4, 3 rounds, Haiku on `FlagGame(modality="image")` (candidates and
  crop as PNG images, no text grids), `Budget(soft_usd=0.40, hard_usd=0.50)`. Not in the
  default `--arms` (run it with `--arms haiku-image`), so the default total stays under the cap.
  Its estimate adds the image overhead measured on the round-1 message (image-mode round message
  minus text-mode round message, in `Provider.estimate_prompt_tokens` tokens: about 730) to the
  3000-token planning prompt and as per-round growth (full memory keeps every round's images).
  `max_calls=5` (not 6) keeps that worst case under the $0.50 ceiling; Haiku averaged 2.9 model
  calls per turn in M2, so the cap does not bind in practice. Its report adds whether posts
  mention visual features (colour words, layout words such as stripe/band/top/left) and whether
  they transcribe the image into colour-letter rows.

- `gemma-image` (WP16): N=4, 3 rounds, `hf:google/gemma-4-26B-A4B-it:deepinfra` (HF_TOKEN) on
  `FlagGame(modality="image")`, broadcast, `max_calls=5`, `Budget(soft_usd=0.40, hard_usd=0.50,
  measurement_usd=0.10)`. The model of `m6_flag_vlm.yaml` (bucket
  `hf://buckets/cmpatino/swarmlab-experiments`). Pricing is the router's listed DeepInfra price
  on 2026-10-07, input 0.07 / output 0.34 USD per M tokens (listing
  `input_modalities: [text, image]`, `supports_tools: true`), cached priced at the input price:
  (0.07, 0.34, 0.07).
- `gemma-manager` (WP16): the same model, N=4, 3 rounds, image mode, the Flag Game paper's
  manager protocol: `FlagGame(blind_agents=1)` (a000 has no crop and no `guess` tool), the
  `star` topology (members' posts reach only a000, a000's reach everyone) and a000 assigned the
  built-in `manager` role; same budget. Its report adds the manager's posts (count and the
  first two), whether the manager tried to guess (it should not; `not_allowed` if it did), and
  how many member posts reached the manager.
  Neither gemma arm is in the default `--arms`; run them with `--arms gemma-image,gemma-manager`.

`--arms` (comma list, default `haiku,qwen,haiku-json`) selects the arms; the total cap applies to
the selected arms.

Per arm it prints: the estimate, spend (swarm, measurement, calls), cost per turn, tool-call
success rate, yield kinds, turn notes (`length`, `text_tool_fallback`), finish reasons, probe ok
rate, accuracy by round (world and probe), and whether the arm's tool protocol worked (at least
one tool call was executed with ok=True and at least one `guess` was accepted, and no response
had unparseable tool arguments).
"""
from __future__ import annotations

import argparse
import copy
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from swarmlab import Board, Budget, Experiment
from swarmlab.metrics.belief import Accuracy, Consensus
from swarmlab.participants import LLMAgent
from swarmlab.probes import BeliefProbe
from swarmlab.providers.openai_compat import OpenAICompatProvider
from swarmlab.roles import assign
from swarmlab.worlds import FlagGame

BUDGET = Budget(soft_usd=0.40, hard_usd=0.50, measurement_usd=0.10)
SMALL_BUDGET = Budget(soft_usd=0.15, hard_usd=0.20, measurement_usd=0.05)
TOTAL_CAP_USD = 1.00  # worst case over all arms must stay under this
QWEN_ID = "Qwen/Qwen3.5-9B:deepinfra"
QWEN_PRICING = (0.10, 0.15, 0.10)  # router-listed DeepInfra price (input, output, cached=input)
HAIKU = "anthropic:claude-haiku-4-5"
GEMMA_ID = "google/gemma-4-26B-A4B-it:deepinfra"
GEMMA_PRICING = (0.07, 0.34, 0.07)  # router-listed DeepInfra price 2026-10-07 (cached=input)
AGENT = {"max_tokens": 1024, "max_calls": 6}
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}
DEFAULT_ARMS = ("haiku", "qwen", "haiku-json")
ARMS: dict[str, dict] = {
    "haiku": {"model": HAIKU, "n": 4, "rounds": 3, "agent": {}},
    "qwen": {"model": f"hf:{QWEN_ID}", "n": 4, "rounds": 3, "agent": {"extra": NO_THINKING}},
    "haiku-json": {"model": HAIKU, "n": 2, "rounds": 2, "agent": {"tool_protocol": "json"},
                   "budget": SMALL_BUDGET},
    "haiku-image": {"model": HAIKU, "n": 4, "rounds": 3, "agent": {"max_calls": 5},
                    "world": {"modality": "image"}},
    "gemma-image": {"model": f"hf:{GEMMA_ID}", "n": 4, "rounds": 3, "agent": {"max_calls": 5},
                    "world": {"modality": "image"}},
    "gemma-manager": {"model": f"hf:{GEMMA_ID}", "n": 4, "rounds": 3, "agent": {"max_calls": 5},
                      "world": {"modality": "image", "blind_agents": 1}, "topology": "star",
                      "manager": True},
}
BASE_PROMPT_TOKENS = 3000  # Experiment.estimate's planning default
COLOUR_WORDS = ("red", "green", "blue", "yellow", "black", "white", "orange", "purple", "cyan",
                "magenta", "brown", "teal", "pink", "grey", "gray")
LAYOUT_WORDS = ("stripe", "band", "block", "top", "bottom", "left", "right", "horizontal",
                "vertical", "upper", "lower", "middle", "corner", "half", "column", "row")
_LETTER_ROW = re.compile(r"(?m)^\s*[rgbykwopcmnt]{3,}\s*$")


def experiment(arm: str) -> Experiment:
    cfg = ARMS[arm]
    participants = [LLMAgent(model=cfg["model"], **{**AGENT, **cfg["agent"]})
                    for _ in range(cfg["n"])]
    if cfg.get("manager"):  # WP16: a000 is the blind manager at the centre of the star
        assign(participants[0], "manager")
    return Experiment(
        name="real-smoke",
        arm=arm,
        world=FlagGame(n_candidates=8, **cfg.get("world", {})),
        participants=participants,
        medium=Board(topology=cfg.get("topology", "broadcast")),
        metrics=["belief.accuracy", "belief.consensus",
                 Accuracy(source="probe:belief"), Consensus(source="probe:belief")],
        probes=[BeliefProbe()],
        budget=cfg.get("budget", BUDGET),
        providers={"hf": OpenAICompatProvider("hf", pricing={QWEN_ID: QWEN_PRICING,
                                                             GEMMA_ID: GEMMA_PRICING})},
    )


def image_overhead_tokens(exp: Experiment) -> int:
    """Estimated prompt tokens the image round-1 message adds over the text-mode one (for the
    last participant group: a member with a crop, never the blind manager)."""
    from swarmlab.prompts_cmd import group_views
    from swarmlab.providers.base import ChatRequest

    def round_tokens(e: Experiment) -> int:
        _, _, _, agent, view = group_views(e, 1)[-1]
        req = ChatRequest(model=agent.model, messages=[agent.round_message(view)])
        return e.provider_for(agent.model)[0].estimate_prompt_tokens(req)

    text = copy.deepcopy(exp)
    text.world = FlagGame(n_candidates=exp.world.n_candidates,
                          blind_agents=getattr(exp.world, "blind_agents", None) or None)
    return max(0, round_tokens(exp) - round_tokens(text))


def estimate(arm: str, exp: Experiment) -> dict:
    calls = {**AGENT, **ARMS[arm]["agent"]}["max_calls"]
    extra = image_overhead_tokens(exp) if getattr(exp.world, "modality", "text") == "image" else 0
    return exp.estimate(seed=1, max_rounds=ARMS[arm]["rounds"], calls_per_turn=calls,
                        prompt_tokens=BASE_PROMPT_TOKENS + extra, prompt_growth=extra)


def visual_mentions(texts: list[str]) -> dict:
    """How many post texts mention colours, layout words, or transcribe colour-letter rows."""
    def has(words: tuple[str, ...], t: str) -> bool:
        return any(re.search(rf"\b{w}", t, re.IGNORECASE) for w in words)

    return {"posts": len(texts),
            "colour_words": sum(1 for t in texts if has(COLOUR_WORDS, t)),
            "layout_words": sum(1 for t in texts if has(LAYOUT_WORDS, t)),
            "letter_rows": sum(1 for t in texts if _LETTER_ROW.search(t)),
            "visual": sum(1 for t in texts if has(COLOUR_WORDS + LAYOUT_WORDS, t))}


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
    if ARMS[arm].get("world", {}).get("modality") == "image":
        posts = [str((e.get("args") or {}).get("text", "")) for e in called if e["tool"] == "post"]
        vm = visual_mentions(posts)
        print(f"  visual features {vm['visual']}/{vm['posts']} posts mention colours or layout "
              f"(colour words {vm['colour_words']}, layout words {vm['layout_words']}, "
              f"colour-letter rows {vm['letter_rows']})")
        for t in posts[:2]:
            print(f"    e.g. {t[:160]!r}")
    if cfg.get("manager"):
        mgr_posts = [str((e.get("args") or {}).get("text", "")) for e in called
                     if e["tool"] == "post" and e["agent"] == "a000"]
        mgr_guess = [e for e in returned if e["agent"] == "a000"
                     and (e["result"] or {}).get("error") == "not_allowed"]
        post_author = {e["post_id"]: e["agent"] for e in events if e["type"] == "post"}
        to_mgr = sum(1 for e in events if e["type"] == "delivery" and e["recipient"] == "a000"
                     and post_author.get(e["post_id"]) != "a000")
        print(f"  manager        {len(mgr_posts)} posts, {len(mgr_guess)} not_allowed calls, "
              f"{to_mgr} member posts delivered to it")
        for t in mgr_posts[:2]:
            print(f"    e.g. {t[:200]!r}")
    errs = [e for e in turns if e.get("error")]
    if errs:
        print(f"  first turn error: {errs[0]['error'][-400:]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("runs/real_smoke"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                    help=f"comma-separated arms out of {', '.join(ARMS)}")
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown or not arms:
        print(f"unknown arms {unknown}; choose from {list(ARMS)}")
        return 2
    exps = {arm: experiment(arm) for arm in arms}
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
