# WP-minesweeper handoff: Minesweeper world

Branch `minesweeper-world`. A built-in world contributed from an experiment workspace, not from a milestone of the build plan; no contract changes. `ruff check swarmlab tests examples tools` is clean and `pytest -q` passes (39 tests in `tests/test_minesweeper.py`).

## What landed

| file | change |
|---|---|
| `swarmlab/world/minesweeper.py` (new) | `Minesweeper` (entry point `minesweeper`; no `requires`; `begin_round` deduces the provable cells, `end_round` returns the round's tally as `round_resolved`, `report_sections` adds an outcome / mines / allocation table, `restore` refuses a different board size or mine count); metrics `Cleared`, `Collisions`, `ProductiveRate`, `AllocationEfficiency`, `MineHits` (`minesweeper.*`); fake scripts `fake_random` (`minesweeper_random`) and `fake_allocator` (`minesweeper_allocator`); pure helpers `deduce`, `neighbours`, `column_labels`, `parse_observation` |
| `pyproject.toml` | the world, the five metrics and the two fake scripts as entry points |
| `examples/minesweeper.yaml` (new) | `random` and `allocator` arms on the fake scripts (4 agents, beginner board, `on_mine: continue`, `commit: round_end`, push broadcast), the rounds sentence in `system_prompt_append`, LLM arm settings in comments |
| `tests/test_minesweeper.py` (new) | parameter validation, seeded reset and safe flood-filled opening, the three classic difficulties (reset, observation round trip, sizes), the information boundary (public observation, no minefield, task text without round mechanics), reveal / flood / win, `on_mine` end and continue, shared unchecked flags and refused reveals keeping the budget, the per-round limit at validate and commit (and with `free_flags: false`), collisions vs stale reveals by what the agent had seen (unit level, and through runs under `round_end` and `immediate` commit), no action after a win or loss in the same batch, a board cleared by the opening (no play, no ideal work), `deduce` (single-clue, subset, flags, exploded mines; soundness on played boards of every difficulty), mine-hit classification, `end_round` tallies and the metrics folding them equal to `score()` every round, snapshot round trip and the size check, `render_state`, entry points, the example spec end to end with replay, allocator vs random collisions, a standard-rule Python API run that replays, the report section, forks (changed rule continues, changed size refused) |
| `docs/guide/minesweeper.md` (new), `docs/README.md` | guide page and its index line |

No other module changed.

## Behaviour and choices

- The task text describes the game only (agents, board, rules, win condition), as for Hidden Sites: how rounds work and any goal beyond winning (speed) belong in `system_prompt_append`.
- Flags are shared and never checked, on purpose: they are a second, stigmergic channel between agents (the only one in a no-board arm). A refused action (flagged cell, not hidden, not flagged) does not use up the agent's per-round budget.
- `deduce` treats flagged cells as unknown and empty cells as 0 clues, so a wrong flag next to an empty cell is provably safe. It is sound, not complete; the accounting it drives (`ideal_reveals`, certain vs guess reveals, provable / unforced / forced mine hits) is therefore a lower bound on what agents could know, and is judged against the round-start board under either commit mode.
- A repeat reveal is a collision only when another agent's reveal opened the cell after this agent last observed the board: `observe` records a reveal counter per agent (the runner observes at every turn start), so `round_end` repeats are collisions and `immediate` repeats of visible cells are stale without the world knowing the commit mode. That counter is the only state `observe` touches, and it never changes what an agent is shown.
- No action is accepted once the game is won or lost, even later in the batch that ended it (review of PR #21: a win then a mine in one batch used to score both). A round that starts with the game over counts 0 ideal reveals.
- `n_agents` in the accounting is the agents given at reset; agents killed by an intervention are not subtracted (the `productive_rate` metric uses the same count).
- `reset` always draws the opening shuffle, so `safe_opening: false` gives the same minefield as `true` for a seed.

## Where it came from

`/data/workspaces/minesweeper` (2026-10-08/09): can agents allocate parallel work through the message board to clear a board in fewer rounds? Arms: one agent, four agents without the board, four with it, and four with a one-sentence nudge to split the work; 12 x 12 with 22 mines, `on_mine: continue`, Qwen3.8-27B on DeepInfra with thinking on. Only the board arm's first seed has run (cleared in round 36, $6.23, 3.2 h; agents posted explicit allocation plans from round 2, 8 collisions in 89 reveals, 276 flags and 260 unflags, 6 mine hits of which 4 were cells the board proved to be mines). The workspace's draft world differs from this one: single-clue deduction only, `on_mine: continue` by default, a 12 x 12 default board, the task text stated the round mechanics and a speed goal, and its metrics re-derived the tallies from `action_committed` feedback instead of folding `round_resolved`.

## Open points

- `deduce` is incomplete: exact inference over the frontier (enumerating mine layouts consistent with the board and the mine count) would also give each hidden cell's mine probability, and with it "was the agent's guess the safest one"; it is exponential in the frontier in the worst case, so it is not run every round.
- Thinking models spend most of a round generating (about 5k tokens per call on Qwen3.8-27B), and agents churn flags; both make LLM runs on large boards long and costly (see the guide's Cost section).
