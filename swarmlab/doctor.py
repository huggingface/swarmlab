"""`swarmlab doctor`: is this machine ready to run an experiment?

`checks(specs=(), offline=False, env=None) -> list[Check]`; `report(checks) -> str` is the
one-screen text; the CLI exits 1 when any check has status `fail`.

Checks, in order:

- `python`: the interpreter satisfies `requires-python` (>= 3.12).
- `spec:<path>`: each given spec validates and every arm builds (unknown plugins, unpriced models).
- `extra:anthropic`: the `anthropic` package imports. `fail` when a given spec uses an
  `anthropic:` model, else `warn` (only needed for Claude models).
- `key:<provider>`: `ANTHROPIC_API_KEY`/`ANTHROPIC_KEY`, `HF_TOKEN`, `OPENAI_API_KEY` presence
  (values are never printed). Missing is `fail` when a given spec uses that prefix, else `info`.
- `reach:hf`: one GET of the HF router models listing (also refreshes the model catalog cache).
- `reach:anthropic`: one GET of `https://api.anthropic.com/v1/models?limit=1` with the key
  (`skip` without a key). A 401/403 is `fail` (bad key).
  Unreachable is `fail` when a spec uses that prefix, else `warn`. Both are `skip` with `--offline`.
- `git`: the commit and dirty flag of the swarmlab checkout, as recorded in every run. Not a git
  checkout (e.g. installed from a wheel) is `warn`: runs then record commit "unknown".
"""
from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

Status = Literal["ok", "info", "warn", "fail", "skip"]

KEYS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_KEY"),
    "hf": ("HF_TOKEN",),
    "openai": ("OPENAI_API_KEY",),
}
ANTHROPIC_MODELS_URL = "https://api.anthropic.com/v1/models?limit=1"
MIN_PYTHON = (3, 12)


@dataclass
class Check:
    name: str
    status: Status
    detail: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def _spec_prefixes(spec: Path) -> tuple[set[str], list[Check]]:
    """Model prefixes used by a spec, and a `spec:` check (building every arm)."""
    from .experiment import Experiment
    from .spec import load_experiment_yaml

    prefixes: set[str] = set()
    try:
        doc = load_experiment_yaml(spec)
        for arm in doc["arms"].values():
            models = [(g.get("params") or {}).get("model") for g in arm["participants"]]
            models += [(p.get("params") or {}).get("coder_model") for p in arm["probes"]]
            prefixes |= {m.split(":", 1)[0] for m in models if isinstance(m, str) and ":" in m}
    except Exception as e:  # noqa: BLE001
        return prefixes, [Check(f"spec:{spec}", "fail", f"{type(e).__name__}: {e}")]
    try:
        for arm in doc["arms"]:
            Experiment.from_yaml(spec, arm)
    except Exception as e:  # noqa: BLE001
        return prefixes, [Check(f"spec:{spec}", "fail", f"{type(e).__name__}: {e}")]
    used = ", ".join(sorted(prefixes)) or "no model providers"
    return prefixes, [Check(f"spec:{spec}", "ok", f"{len(doc['arms'])} arm(s) build; uses {used}")]


def _get(url: str, headers: Mapping[str, str] | None = None) -> tuple[int | None, str]:
    import httpx

    try:
        resp = httpx.get(url, headers=dict(headers or {}), timeout=10.0)
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"
    return resp.status_code, resp.text[:200]


def checks(specs: Iterable[Path | str] = (), offline: bool = False,
           env: Mapping[str, str] | None = None) -> list[Check]:
    env = os.environ if env is None else env
    out: list[Check] = []

    v = sys.version_info
    out.append(Check("python", "ok" if v[:2] >= MIN_PYTHON else "fail",
                     f"{v.major}.{v.minor}.{v.micro} (needs >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]})"))

    used: set[str] = set()
    for spec in specs:
        prefixes, spec_checks = _spec_prefixes(Path(spec))
        used |= prefixes
        out += spec_checks

    try:
        import anthropic

        out.append(Check("extra:anthropic", "ok", f"anthropic {anthropic.__version__} installed"))
    except ImportError:
        needed = "anthropic" in used
        out.append(Check("extra:anthropic", "fail" if needed else "warn",
                         "not installed; needed for anthropic: models: "
                         "`uv sync --extra anthropic` or `pip install -e .[anthropic]`"))

    present: dict[str, str | None] = {}
    for prefix, names in KEYS.items():
        found = next((n for n in names if env.get(n)), None)
        present[prefix] = found
        if found:
            out.append(Check(f"key:{prefix}", "ok", f"{found} is set"))
        else:
            out.append(Check(f"key:{prefix}", "fail" if prefix in used else "info",
                             f"{' or '.join(names)} not set" + (" (a spec uses it)" if prefix in used
                                                                else "")))

    if offline:
        out.append(Check("reach:hf", "skip", "--offline"))
        out.append(Check("reach:anthropic", "skip", "--offline"))
    else:
        from .providers.catalog import ROUTER_URL, fetch_router, write_cache

        try:
            data = fetch_router(timeout_s=10.0)
            write_cache(data)
            out.append(Check("reach:hf", "ok",
                             f"{ROUTER_URL}: {len(data['data'])} models listed (catalog cached)"))
        except Exception as e:  # noqa: BLE001
            out.append(Check("reach:hf", "fail" if "hf" in used else "warn",
                             f"{ROUTER_URL}: {type(e).__name__}: {e}"))
        key_env = present["anthropic"]
        if not key_env:
            out.append(Check("reach:anthropic", "skip", "no Anthropic key"))
        else:
            status, text = _get(ANTHROPIC_MODELS_URL, {"x-api-key": env[key_env],
                                                       "anthropic-version": "2023-06-01"})
            if status == 200:
                out.append(Check("reach:anthropic", "ok", "models endpoint answered with the key"))
            elif status in (401, 403):
                out.append(Check("reach:anthropic", "fail", f"HTTP {status}: key rejected"))
            else:
                out.append(Check("reach:anthropic", "fail" if "anthropic" in used else "warn",
                                 f"{ANTHROPIC_MODELS_URL}: {status or text}"))

    from .spec import git_identity

    repo = Path(__file__).resolve().parent.parent
    commit, dirty = git_identity(repo)
    if commit == "unknown":
        out.append(Check("git", "warn", f"{repo} is not a git checkout; runs record commit 'unknown'"))
    else:
        out.append(Check("git", "warn" if dirty else "ok",
                         f"{commit[:12]}{' dirty (uncommitted changes to tracked files)' if dirty else ' clean'}"))
    return out


def ok(results: Iterable[Check]) -> bool:
    return all(c.status != "fail" for c in results)


def report(results: list[Check]) -> str:
    width = max(len(c.name) for c in results)
    lines = [f"{c.status.upper():<5} {c.name:<{width}}  {c.detail}" for c in results]
    fails = sum(c.status == "fail" for c in results)
    lines.append("ready" if not fails else f"{fails} problem(s) to fix before running")
    return "\n".join(lines)
