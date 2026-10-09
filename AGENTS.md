# Working in this repo

- `docs/INTERFACE.md` (M1a) and its extensions `INTERFACE-M1b.md`, `INTERFACE-M3a.md`, `INTERFACE-M3c.md`, `INTERFACE-M4.md`, `INTERFACE-M5.md` are the binding contracts. Implement exactly the names, signatures, and semantics they define. Anything it leaves open is your choice; document it in the module docstring.
- `docs/DESIGN.md` explains why. Read the sections relevant to your work package before coding.
- Python 3.12, pydantic v2, asyncio. No global `random`; use `swarmlab.rng.derive`.
- `docs/README.md` indexes the user guide (`docs/guide/`: spec reference, CLI, budgets, Flag Game, analysis); the top-level README is the human overview and holds no reference material.
- `docs/handoff/INDEX.md` says what each work package built; `docs/known-issues.md` lists open problems.
- Tests live in `tests/`; run them with `UV_PROJECT_ENVIRONMENT="$AM_LOCAL/envs/swarmlab" uv run pytest`.
- Commits: plain messages, no `Co-Authored-By` or attribution trailers, do not change git user config.
- Do not edit files owned by another work package; stub what you need and note the stub in your handoff.

# Running experiments

- When the task is to design, smoke, launch, analyse, report on or publish an experiment, follow `skill/SKILL.md`: its first-real-run checklist (doctor, validate, scripted dry run, `swarmlab prompts` review by a second agent, N=4 smoke with a hard ceiling, estimate, launch, fetch, report, publish; under a $2 total, the short version: `prompts` + `diff` and the first arm at N<=6 as the smoke) is the tested workflow, and every rule in it names a command.
- Spending steps need the user's approval of the estimate first; publishing stays private unless the user asks for `--public`.
- Read finished runs through `swarmlab export` tables and sessions or `swarmlab report`, not by parsing `events.jsonl` by hand.
