# Working in this repo

- `docs/INTERFACE.md` is the binding contract for M1a. Implement exactly the names, signatures, and semantics it defines. Anything it leaves open is your choice; document it in the module docstring.
- `docs/DESIGN.md` explains why. Read the sections relevant to your work package before coding.
- Python 3.12, pydantic v2, asyncio. No global `random`; use `swarmlab.rng.derive`.
- Tests live in `tests/`; run them with `UV_PROJECT_ENVIRONMENT="$AM_LOCAL/envs/swarmlab" uv run pytest`.
- Commits: plain messages, no `Co-Authored-By` or attribution trailers, do not change git user config.
- Do not edit files owned by another work package; stub what you need and note the stub in your handoff.
