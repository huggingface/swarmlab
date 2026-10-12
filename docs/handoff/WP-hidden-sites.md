# WP-hidden-sites handoff: HiddenSites world

Branch `hidden-sites-world`. A built-in world contributed from an experiment workspace, not from a milestone of the build plan; no contract changes. `ruff check swarmlab tests examples tools` is clean and `pytest -q` passes (17 tests in `tests/test_hidden_sites.py`).

## What landed

| file | change |
|---|---|
| `swarmlab/world/hidden_sites.py` (new) | `HiddenSites` (entry point `hidden_sites`; `requires` commit `immediate` and `max_rounds == rounds`; `end_round` resolves the round, `report_sections` adds a references table); metrics `ExpectedW`, `ExpectedCum`, `ActualPoints`, `ActualCum` (`hidden.expected_W`, `hidden.expected_cum`, `hidden.actual_points`, `hidden.actual_cum`); fake script `fake_chooser` (`hidden_chooser`); pure helpers `resolve_round`, `trial_success`, `expected_points`, `references`, `optimum_counts`, `site_labels`, `number_word`, `parse_observation`, `parse_agent_id` |
| `pyproject.toml` | the world, the four metrics and the fake script as entry points |
| `examples/hidden_sites.yaml` (new) | `team` and `reminder` arms on `fake:hidden_chooser` (12 agents, `commit: immediate`, `max_rounds: 20`), with the LLM arm settings and the cost note in comments |
| `tests/test_hidden_sites.py` (new) | references and the optimum, matched permutation and trial key per seed, trial stream independent of call counts, one-choice rule, dynamic `choose_site` enum, information boundary of observation, feedback, schema and task text, resolution at `end_round` and metrics folding `round_resolved` equal to `score()` after every round, snapshot round trip, forks (a changed site set rejected at restore; changed `rounds`/`show_history` continue), fake script, entry points, the example spec end to end, a Python API run that replays, the requirements check, the report section |
| `docs/guide/hidden-sites.md` (new), `docs/README.md` | guide page and its index line |

No other module changed.

## Behaviour

The world, metrics and observation format are the experiment's code unchanged: a differential check over 20 seeds x 20 rounds of random choices gave identical observations, tool schemas, task text, feedback, `score()`, `verify()` and `render_state()`. Differences from the source file:

- imports are package-relative and the classes carry `entry_point`s, so specs say `type: hidden_sites` and `hidden.expected_W` instead of `hidden_sites:HiddenSites` / `hidden_sites:ExpectedW` (spec hashes of the experiment's spec differ for that reason only);
- the experiment's reminder sentence (`REMINDER`) is gone from the module; it is a treatment, not part of the task;
- the import-time `assert`s on the default references moved into the tests (no 50k-allocation search on import);
- `fake_chooser` reads the site labels from the `choose_site` schema, so it also works with `n_sites != 8` (identical for 8 sites).
- Rebased onto the core hooks: the `choose_site` schema uses callable `@tool` params (no `tool_schemas` override), rounds are resolved in `end_round` and logged as `round_resolved` (no eager resolution at `begin_round`, no lazy last round in `score()`), the metrics fold that event (no `needs_truth`, no recomputed trials), and `requires` checks `commit` and `max_rounds`. Observations, feedback, scores and metric values are unchanged.

## Where it came from

`/data/workspaces/ideas-market/hidden-sites` (2026-10-08/09): does a one-sentence reminder to communicate about crowding ("Communicate with your teammates to avoid overcrowding sites, where additional agents contribute less to the team.", via `system_prompt_append`) help agents allocate effort when site qualities are hidden? Two arms, `team` (task text only) vs `reminder`, 12 agents on `hf:Qwen/Qwen3.8-27B:ovhcloud`, 20 rounds, `commit: immediate`, window memory of 6 rounds, broadcast board. Seeds 1 and 2 of both arms ran 20 rounds and replay cleanly ($5.4-6.4 per run; one run $2.0). Cumulative expected points over 20 rounds (random reference 36.6, oracle 52.7): team 32.3 / 36.9, reminder 39.0 / 22.0. Every swarm scouted all sites in round 1, pooled per-site tallies on the board and consolidated on one or two sites by round 5-7; which sites survived depended on a few early samples (premature abandonment of the 0.35 and 0.25 sites after two or three failures), and in one run all 12 agents ended on the best site. The dry arms on the fake script ran seeds 1-3. That workspace holds the analysis and replay-page scripts; they are not part of this package.

## Open points

- The LLM round message starts with the harness's `Round r.` line followed by the world's `Round r of R.` line (redundant, harmless).
- Cost is dominated by the board: agents post long tallies of positions, and every later turn reads them.
