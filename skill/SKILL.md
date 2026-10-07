---
name: swarmlab
description: Run a swarmlab multi-agent experiment end to end with the swarmlab CLI. Use when asked to design, smoke, launch (locally or on HF Jobs), analyse, report on, or publish a swarmlab experiment, or to resume or fork a swarmlab run.
---

# swarmlab experimenter workflow

A swarmlab experiment runs LLM-agent swarms on a task world in rounds and logs every event. You change the
hypothesis in a YAML spec; the CLI does the execution. Every step below is a command; every
command takes `--json` (one JSON object on stdout) and exits 0 on success, 2 on an invalid spec,
1 otherwise. `swarmlab COMMAND --help` is the reference for flags.

Spend is real money. Each step that spends ends on a check you can see before the next one
starts: a run summary, an estimate, a replay that matches.

## Install

From a clone of the repo (`uv sync --extra anthropic --extra hub` gives the CLI as
`uv run swarmlab`), copy this file into your agent's skill directory:

| agent | user-level path |
|---|---|
| Claude Code | `~/.claude/skills/swarmlab/SKILL.md` |
| Codex | `~/.agents/skills/swarmlab/SKILL.md` |
| OpenCode | `~/.config/opencode/skills/swarmlab/SKILL.md` (it also reads `~/.claude/skills/`) |

```bash
for d in ~/.claude/skills ~/.agents/skills ~/.config/opencode/skills; do
  mkdir -p "$d/swarmlab" && cp skill/SKILL.md "$d/swarmlab/SKILL.md"; done
```
Project-level copies work too: `.claude/skills/swarmlab/`, `.agents/skills/swarmlab/`,
`.opencode/skills/swarmlab/` inside the repo.

## First real run: the checklist

Size it to the budget first. Put the user's total in the spec as `budget: {total_usd: T}` (top
level): `swarmlab run` then never starts a run that could push the experiment past T.

- **Total under $2: the short checklist.** A separate smoke run and a second-agent review would
  eat a large share of the money, and one round's worst-case estimate can be a fifth of the
  total, so size the caps from measured spend, not from the estimate:
  1. `swarmlab doctor SPEC.yaml` and `swarmlab validate SPEC.yaml` (items 1-2 below).
  2. Dry run with `fake:reader` (item 3; free): checks the metrics and the replay.
  3. Prompt review without a second agent: `swarmlab prompts SPEC.yaml --arm A > A.txt` for
     every arm, then `diff A.txt B.txt` -> only the manipulated text differs, no correctness
     hints, every needed tool named.
  4. Caps go in each arm's own `budget:` (`arms.A.budget: {soft_usd: 0, hard_usd: X}`), not the
     top level: budgets are part of the spec hash, so changing a top-level cap later would make
     the finished first arm look like a different run. Keep only `total_usd: T` at the top.
  5. First real arm = smoke = data, at N <= 6 agents, with `hard_usd` = 50% of the total and no
     soft cap (`soft_usd: 0`: a soft cap at this size stops the run a round early):
     `swarmlab run SPEC.yaml --arm FIRST --seed S` -> `end=max_rounds`. Note its actual spend
     (`spend=$X` on the status line, `spend_usd` under `--json`) and X / rounds = its measured
     per-round cost. The run is kept as the arm's data, not repeated.
  6. Set the second arm's `hard_usd` from that measurement: about 1.5 x the first arm's actual
     `spend_usd` (same model and N, one variable changed), and no more than T minus that spend;
     `soft_usd: 0`. `swarmlab run SPEC.yaml` then skips the finished first arm and prints an
     `existing:` line with its actual spend and per-round cost; `total_usd` counts that spend
     (including anything added by `resume`) before starting the second arm.
  7. A `hard_ceiling` end (round in flight discarded) or `hard_ceiling_probes` (last round kept,
     some probes skipped): `swarmlab resume RUN_DIR --add-budget D` if T minus the spend so far
     allows (resume itself does not check `total_usd`), else report the rounds you have.
  8. `swarmlab report runs/ --out report.md`; it lists skipped probes per arm.
- **Total of $2 or more: the full checklist below**, all eleven items.

Do these in order. Each item is done when its check holds; a failing check is fixed before the
next item.

1. **Environment.** `swarmlab doctor SPEC.yaml` -> exit 0 (keys, extras, provider reachability,
   clean git). A new experiment starts from `swarmlab init NAME` (writes `NAME.yaml`, `NAME.py`)
   in a project directory outside the swarmlab checkout (`uv run --project <repo> swarmlab ...`
   from there); `swarmlab run` refuses to write `runs/` inside the checkout unless `--out` is
   given.
2. **Spec.** `swarmlab validate SPEC.yaml` -> every arm resolves; its table shows each arm's
   agents, models, soft/hard/measurement caps, probes and metrics: check them against the plan.
   `swarmlab spec-reference` lists every YAML key and its shape (e.g. `medium: {topology:
   {type: gossip, params: {k: 1}}}`, per-arm `budget:`); a shape error names the key and the
   shape expected there.
   `swarmlab metrics` lists every metric name with what it measures (e.g. `comm.post_rate`,
   `comm.posts_per_round` per agent).
   Model ids come from
   `swarmlab models --tools --search QWEN` (prices are looked up, never typed). Every arm that
   calls a paid model has `budget: {hard_usd: X}`; `hard_usd: 0` means no ceiling.
3. **Dry run with scripted agents.** Copy the arm into a dry-run arm whose participants are
   `{type: evidence_aggregator, count: N}` or `{type: llm, params: {model: "fake:reader"}}`, then
   `swarmlab run SPEC.yaml --arm dry --seed 0 --yes` -> `status=ended`, metrics present, no spend.
   `swarmlab replay RUN_DIR` -> exit 0.
4. **Prompts review by a second agent.** `swarmlab prompts SPEC.yaml --arm A > prompts-A.txt` for
   every arm (no model call). Hand the files to a reviewer agent that did not write the spec; it
   checks that each arm's system prompt and round-1 message differ only in the manipulated
   variable, carry no correctness hints, and name every tool the arm needs. Fix the spec and
   re-render until the reviewer signs off.
5. **Smoke at N=4 with a hard ceiling.** A smoke arm: the real model, `count: 4`,
   `options: {max_rounds: 3}`, `budget: {hard_usd: 0.50, measurement_usd: 0.10}`.
   `swarmlab run SPEC.yaml --arm smoke --seed 0` -> `end=max_rounds` (not `hard_ceiling`), then
   `swarmlab report RUNS_DIR` -> tool protocol health shows no errored turns and few `length`
   / `max_tokens` finishes (truncated replies). A `hard_ceiling` end (the round in flight was
   discarded) or `hard_ceiling_probes` (the last round was kept, some of its probes were
   skipped) means raise the budget or cut rounds; `swarmlab resume RUN_DIR --add-budget D`
   continues the same run with D more dollars (`--budget-hard X` sets the run's
   total, spend so far and discarded rounds included).
6. **Estimate.** `swarmlab estimate SPEC.yaml --prompt-growth TOKENS --calls-per-turn C` -> total
   for every arm x seed. Take the growth and calls per turn from the smoke run's `inference` table
   (`swarmlab export RUN_DIR`); without them the worst case (each group's `max_calls`, else the
   runner cap) overstates real spend 4-6x. The user approves
   the total before launch.
7. **Launch.** Local: `swarmlab run SPEC.yaml` (every arm x seed; asks before spending; already
   finished runs are skipped). Self-hosted model on HF Jobs (spec models `vllm:<model>`):
   `swarmlab job run SPEC.yaml --model ORG/MODEL` prints the plan and estimate; add `--launch`
   after approval.
8. **Watch.** Local runs print a summary per run. Jobs: `swarmlab job status JOB_ID`,
   `swarmlab job logs JOB_ID --follow`.
9. **Fetch** (jobs): `swarmlab job fetch RUN_ID --out runs/` for each run id the log names ->
   `swarmlab replay runs/RUN_ID` exits 0.
10. **Report.** `swarmlab report runs/ --out report.md` -> one row per arm (dry runs on `fake:`
    models are listed as simulated and kept out of the tables and spend); read it before
    claiming any effect, and say how many seeds stand behind each number.
11. **Publish.** `swarmlab publish runs/ --repo OWNER/EXPERIMENT` -> private dataset repo with
    tables, sessions, raw logs and a card. `swarmlab view RUN_DIR --publish OWNER/EXPERIMENT`
    adds the replay page. `--public` (makes the repo public and tags it for the Hub's trace
    viewer) only when the user asks for it, after they have read the traces.

## Rules

- **Change the spec, never the code, to change the hypothesis.** One arm per condition; arms of
  one experiment differ only in the manipulated variable. `swarmlab prompts` shows what differs.
- **Change prompts through params, then diff.** An `llm` group's `system_prompt_append: "..."`
  adds text after the default prompt's task section (before the tool list); `system_prompt`
  replaces the whole Jinja2 template (variables `agent`, `role`, `description`, `tools`,
  `system_prompt_append`; `file:<path>` works). Prefer the append: copying the template invites
  accidental differences. `swarmlab prompts SPEC.yaml --arm A > A.txt` per arm, then `diff`, must
  show only the manipulated text. The default prompt already lists every tool (`read_board`,
  `post`), so a "mention the board" arm is a nudge, not awareness.
- **Resume or fork instead of rerunning.** A crashed or budget-ended run continues with
  `swarmlab resume RUN_DIR [--add-budget D]`; a counterfactual from round R is
  `swarmlab fork RUN_DIR --at R --spec edited.yaml`. Both reuse the inference cache, so the
  shared prefix costs nothing.
- **Re-running a finished run is a skip.** `swarmlab run` skips run dirs with the same spec hash;
  `--rerun` writes `RUN_ID__rN` beside it when you need a repeat.
- **Replay before you trust a run.** `swarmlab replay RUN_DIR` recomputes metrics and score from
  the log with zero provider calls and exits 1 on any mismatch.
- **Analyse from exports.** `swarmlab export RUN_DIR` writes `export/tables/<family>.parquet`
  (turns, tool_calls, posts, deliveries, reads, actions, inference, probes, metrics,
  interventions, rounds, run, other, discarded_inference; keys `experiment, arm, seed, run, round,
  agent`; `run.json` has `spend` from the ledger and `spend_discarded_usd`) and one pi
  session per agent in `export/sessions/` for reading a conversation turn by turn.
- **Look at one run by eye** with `swarmlab view RUN_DIR` (writes `view.html`, no server).
- **Restore a published run** with `swarmlab fetch-published OWNER/EXPERIMENT RUN_ID --out runs/`;
  it replays like a local one.
- **Slow providers and sequential commit do not mix**: `commit: immediate` puts every call on the
  critical path; keep it to fast providers or small N (`swarmlab estimate` does not model this).
- **Reasoning models need room**: set `max_tokens: 2048` or turn thinking off through
  `params: {extra: {chat_template_kwargs: {enable_thinking: false}}}`; `swarmlab report` counts
  `length` and `max_tokens` finishes (truncated replies).
