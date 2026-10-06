# swarmlab

A scientific testbed for finding the primitives that make heterogeneous groups of LLM agents collaborate well or badly. Working name; see `docs/DESIGN.md` for the design and `docs/INTERFACE.md` for the M1a contract.

Status: M1a in progress. Nothing here calls a model yet.

## Development

```
UV_PROJECT_ENVIRONMENT="$AM_LOCAL/envs/swarmlab" uv sync --extra dev
UV_PROJECT_ENVIRONMENT="$AM_LOCAL/envs/swarmlab" uv run pytest
```
