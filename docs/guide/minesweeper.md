# Minesweeper

A shared-board collaboration world: the whole team plays one game of Minesweeper, so agents must split the work (two agents revealing the same cell waste a reveal), share deductions and avoid cells the board already proves are mines. The board size and mine count set the difficulty. Module: `swarmlab/world/minesweeper.py` (its docstring is the exact reference). Example spec: `examples/minesweeper.yaml`. Back to the [docs index](../README.md).

## The task

`reset` hides `mines` mines in a `height` x `width` board (a seeded sample of the cells) and, by default, opens a safe start: a cell with no neighbouring mines, flood-filled. Every agent sees the whole board. On its turn an agent may `reveal(x, y)` (a 0 opens its neighbours too), `flag(x, y)` a cell it believes is a mine and `unflag(x, y)`; each agent may reveal at most `reveals_per_round` cells per round, and flagging is free unless `free_flags: false`. Flags are shared board state: everyone sees them, anyone may remove one, a flagged cell cannot be revealed until it is unflagged, and nothing says whether a flag is right. The team wins when every safe cell is revealed. Revealing a mine loses the game (`on_mine: end`, the standard rule) or, with `on_mine: continue`, shows `*` and play goes on, so every game can be cleared and the round it was cleared in is always defined.

Same seed, same minefield and opening in every arm, so arms are compared game by game.

The world has no requirements: any commit mode, scheduler, topology and agent count is a legitimate variation, so its task text (`description()`) says nothing about turns, their order, when actions apply or who reads a post. It states the agents (count and ids), the board, the rules above and the win condition. Tell agents how your experiment's rounds work with `system_prompt_append` (see [Changing what agents are told](building-experiments.md#changing-what-agents-are-told)); `examples/minesweeper.yaml` does this for `commit: round_end` with posts pushed to everyone. Add any goal beyond winning (for example "in as few rounds as possible") the same way.

Under `commit: round_end` every agent chooses from the same board and actions apply together in the seeded commit order, so a cell revealed by two agents in one round is a **collision** (`same_round: true` in the second reveal's feedback). Under `commit: immediate` each turn sees the earlier turns' results and a repeat reveal is a stale one.

## Parameters

`world: {type: minesweeper, params: {...}}`

| param | default | meaning |
|---|---|---|
| `height`, `width` | 9, 9 | rows and columns, each 1..99 |
| `mines` | 10 | 1 .. height * width - 1 |
| `on_mine` | `end` | `end`: a mine loses the game and the run ends; `continue`: the cell shows `*`, the hit is counted, play goes on |
| `reveals_per_round` | 1 | reveals per agent per round |
| `free_flags` | true | flags and unflags do not count toward `reveals_per_round` |
| `safe_opening` | true | open a safe, flood-filled start at reset |

### Difficulty

The classic boards, with how many rounds an idealised team needs (median over 30 boards, `on_mine: continue`, one reveal per agent per round). The coordinated team reveals distinct provably safe cells; the random team has each agent pick a provably safe cell independently, so some picks collide; with nothing provably safe, one agent guesses. Speedups are against one agent.

| board | `height`, `width`, `mines` | 1 agent | 4 agents: coordinated / random | 8 agents: coordinated / random |
|---|---|---|---|---|
| beginner | 9, 9, 10 | 18 | 6 / 8 (3.0x / 2.3x) | 4 / 6 (4.5x / 3.3x) |
| intermediate | 16, 16, 40 | 80 | 23 / 26 (3.5x / 3.1x) | 14 / 18 (5.7x / 4.5x) |
| expert | 16, 30, 99 | 203 | 61 / 69 (3.3x / 2.9x) | 40 / 48 (5.1x / 4.3x) |

The ceiling on parallel speedup is set by how many cells are provably safe at once, so returns shrink with more agents; and random picks rarely collide when many cells are safe, so the gap between coordinated and uncoordinated teams is modest for idealised players. Real agents that read the same board tend to pick the same "obvious" cell, which is what the board and its message channel can fix.

## Information boundary

The observation is one text part, identical for every agent:

    Board (12 columns x 9 rows; x = column 0-11, y = row 0-8; # hidden, F flagged, . empty, 1-8 adjacent mines, * exploded mine):
                11
      012345678901
    0 ############
    1 ############
    2 ###22111####
    3 ###1...12###
    4 ###11...1###
    5 ####1...112#
    6 ####1.....1#
    7 ####1211..11
    8 #######1....

    Mines: 15  Flags: 0  Exploded: 0  Hidden: 64

`parse_observation` inverts it. Feedback reports what happened (`opened` with `value` and `cells_opened`, `exploded`, `already_revealed` with `same_round`, `flagged`, `unflagged`, or an error: `flagged`, `not_hidden`, `not_flagged`, `out_of_bounds`, `reveals_per_round`, `game_over`). Agents never see the minefield, whether a flag is right, the provable cells or any accounting below. There are no status tools: the board is public.

## Provable cells and accounting

At the start of every round the world runs `deduce` on the visible board: iterated single-clue rules plus the pairwise subset rule (the 1-2-1 pattern), ignoring flags. It is sound but not complete (exact inference can prove more), so it gives a lower bound on what was knowable. From it, per round:

- `ideal_reveals`: what a perfectly coordinated team could open without guessing, `min(agents x reveals_per_round, provably safe cells)`, or 1 (a forced guess) when nothing is provably safe;
- each reveal that opened a cell is **certain** (provably safe at round start) or a **guess**;
- each mine hit is **provable** (the board proved it was a mine: a reasoning error), **unforced** (provably safe cells were available) or **forced** (nothing was provably safe).

Under `commit: immediate` later turns see more than the round-start board, so these are judged against the round-start board there too (a lower bound).

The world's `end_round` hook returns the round's tally and the board's progress, logged as a `round_resolved` event after the round's `action_committed` events; agents never see it. `score()` adds the run's outcome: `won`, `lost`, `cleared` (share of safe cells revealed), `cleared_round` (the round the last safe cell opened), `first_mine_round` (the round the game is lost under the standard rule, also reported under `on_mine: continue`), `allocation_efficiency` (productive reveals / ideal reveals), `correct_flags`, `wrong_flags` and the cumulative counters.

## Metrics

All five fold the `round_resolved` events, so `swarmlab replay` reproduces them.

| metric | value | denominator |
|---|---|---|
| `minesweeper.cleared` | share of safe cells revealed | safe cells |
| `minesweeper.collisions` | cumulative same-round repeat reveals | reveals |
| `minesweeper.productive_rate` | reveals that opened a cell per agent-round | agent-rounds |
| `minesweeper.allocation_efficiency` | productive reveals / ideal reveals | ideal reveals |
| `minesweeper.mine_hits` | cumulative mines revealed | reveals |

`swarmlab report` adds a table after the Final score table with each run's outcome, `cleared`, `cleared_round`, `first_mine_round`, mine hits (total and provable / unforced / forced), collisions and allocation efficiency. The replay page's world panel (`render_state`) draws the board with the hidden mines as `M`.

## Forks and resumes

The snapshot carries game state only (minefield, board, counters); constructor params come from the spec the fork or resume runs under. A fork may change `on_mine`, `reveals_per_round` and `free_flags`; restoring a snapshot whose board size or mine count differs from the configured one raises `ValueError` before any turn.

## Fake scripts

Two free, deterministic plumbing checks that double as references: `fake:minesweeper_random` reveals a random provably safe cell each turn (a random cell not proven to be a mine when there is none); `fake:minesweeper_allocator` makes agent aNNN reveal the NNN-th provably safe cell in row-major order and posts its pick. Neither flags. On the example's three beginner boards the allocator clears in no more rounds than the random script and collides about half as often.

## Cost

One measured run (2026-10-09): 4 agents on `hf:Qwen/Qwen3.8-27B:deepinfra` with thinking on (`max_tokens: 8192`, `max_calls: 3`, `memory: window`, `window_rounds: 3`), a 12 x 12 board with 22 mines, `on_mine: continue`, `commit: round_end`, broadcast board with push delivery: cleared in 36 rounds, $6.23, 3.2 hours. Time per round is almost all generation: each call wrote about 5,100 completion tokens at about 66 tokens/s (about 80 s), turns used all 3 calls, and a round waits for its slowest agent (about 5 minutes). Agents spent most tool calls on flags (521 flags, 573 unflags against 109 reveals). Budget a hard cap of about $0.17 per round for this setup, raise the provider `timeout_s` to 240 (thinking calls ran up to 75 s; the default is 90), and expect fewer rounds and lower cost on smaller boards.
