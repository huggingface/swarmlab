# WP2 handoff: world (FlagGame text variant)

## What landed

- `swarmlab/world/flaggame.py`: `FlagGame(World)`, `entry_point = "flaggame"`, constructor kwargs exactly as INTERFACE.md §8 (`height=8, width=12, palette=6, n_candidates=8, rival_similarity=0.85, crop_h=3, crop_w=4, candidate_names="letters", status_tools=("my_status","collective_status"), guess_limit=None`). Module-level helpers for scripted participants.
- `tests/test_flaggame.py`: 34 tests (determinism, rival fraction, distinct candidates, crop containment, rival sometimes consistent, observe round-trip, guess/status/score, guess limit, disabled status tools, spec, snapshot/restore, 20-round oracle no-leak).
- `swarmlab/world/base.py`: only change is removing an unused `from typing import Any` so `ruff check swarmlab tests` is clean. No public names touched.

## Observation format

One text part, `\n`-separated:

```
Candidate flags:

A:
<height rows of colour letters, no separators>

B:
...

Your crop:
<crop_h rows of crop_w colour letters>
```

Candidates are listed in name order. Colours are lowercase letters, the first `palette` of `"rgbykwopcmnt"` (so `palette <= 12`), and never collide with candidate names. A header is any line ending in `:`. `observation.private == {"crop_y": y, "crop_x": x}`.

## Helper signatures (import from `swarmlab.world.flaggame`)

```python
parse_observation(text: str) -> tuple[dict[str, list[str]], list[str]]   # (candidates, crop)
candidates_containing(candidates: dict[str, list[str]], crop: list[str]) -> list[str]
contains(grid: list[str], crop: list[str]) -> bool
FlagGame.crop_rows(agent) -> list[str]   # evaluator/test convenience
```

`parse_observation(world.observe(a).parts[0].text)` returns exactly `world.candidates` and `world.crop_rows(a)`. The truth is always in `candidates_containing(...)`. With default settings (twin pairs, post-review) the rival is also consistent with about 53% of crops, and about 67% of crops match more than one candidate.

## Decisions where the contract was silent

- **Per-agent crop streams.** `reset` gets only the world rng. After generating the flags it draws `s_i = rng.getrandbits(64)` once per agent, in `agents` order. Agent i's crop then comes from `derive(s_i, "private", agent_i)`, so crops depend only on (world rng, agent list). The contract's `("private", agent)` root therefore sits under a world-derived seed, not the run seed.
- **Structured flags.** Layouts are horizontal stripes (2-4 bands), vertical stripes (2-4), 2x2 blocks and 2x3 blocks. Neighbouring bands and blocks never share a colour.
- **Rival (changed after the M1a review, finding A1).** `rival_similarity` is gone. Candidates are `n_candidates // 2` twin pairs: a structured flag plus a variant with `rival_edits` (default 1) whole bands/blocks recoloured to a colour that differs from the band's own and its neighbours', so both members are clean structured flags. The truth is a random member of a random pair; the rival is its twin. `n_candidates` must be even and `palette >= 3`. A crop-free "nearest pair, cleaner member" heuristic now scores about chance (test `test_no_crop_free_shortcut`).
- **Distinctness.** Pairs are redrawn until all candidates differ. A ValueError is raised after 1000 attempts.
- **Naming.** Candidates are shuffled and then named ("letters": A..Z, max 26; "numbers": 1..n), so the truth's name is uniform.
- **Validation.** `validate` calls super, then rejects unknown candidates, agents not passed to `reset`, and guesses past `guess_limit`. The `guess` method re-checks the same conditions and returns `accepted=False, feedback={"error": ...}`, because several guesses buffered in one round are validated against round-start state.
- **`collective_status`.** `guess_counts` always lists every candidate name in name order, zeros included, so the key set carries no information. `my_status` for an agent with no state returns `{"current_guess": None, "guesses_made": 0}`.
- **`score()["accuracy"]`.** Divides by all agents given to `reset` (0.0 if there are none).
- **Snapshots** hold game state only. Constructor config attributes are in `_skip_in_snapshot`, so a restored or forked world keeps its own constructor config (a fork can change `guess_limit` or `status_tools`). All state is plain Python data (lists, dicts, str, int, tuples).
