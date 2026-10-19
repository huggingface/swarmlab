# Vendored mini-swe-agent 2.4.6

The `mini_swe_agent` participant (`swarmlab/participants/minisweagent.py`) runs mini-swe-agent's
own agent loop. Rather than depend on the package (which requires litellm and, through it, pins
huggingface-hub<2 and filelock<4), swarmlab carries a copy of the few files the participant
imports.

- Origin: the `mini-swe-agent==2.4.6` wheel from PyPI
  (`mini_swe_agent-2.4.6-py3-none-any.whl`), <https://github.com/SWE-agent/mini-swe-agent>.
- License: MIT, Copyright (c) 2025 Kilian A. Lieret and Carlos E. Jimenez. `LICENSE.md` is a
  byte-for-byte copy of the wheel's `mini_swe_agent-2.4.6.dist-info/licenses/LICENSE.md`.

## Files

The hashes are the `sha256=` values of the wheel's `mini_swe_agent-2.4.6.dist-info/RECORD`
(urlsafe base64 of the SHA-256 digest, no padding) for the original files, before the changes
below.

| Here | Wheel path | RECORD sha256 |
| --- | --- | --- |
| `__init__.py` | `minisweagent/__init__.py` | `CnCDCLG2LMAgtgeZw9VPZlhF95Wz8MT9rHh8lqONEsc` |
| `exceptions.py` | `minisweagent/exceptions.py` | `BZA5PFa-6HPHmmkdy08VyznIXxZYWYuEv7KVv95Wkh0` |
| `serialize.py` | `minisweagent/utils/serialize.py` | `MDWyfP7_35ERfHK8gTE0C9nF9nowELrrTs1rDHQG9GI` |
| `openai_multimodal.py` | `minisweagent/models/utils/openai_multimodal.py` | `xJjsLpfAv0XCEqEN6fXRstJNTeRqBf0Orql1wavj-B8` |
| `actions_toolcall.py` | `minisweagent/models/utils/actions_toolcall.py` | `R7ZmQS5Pg4UItQAJzzTH9j_hXp5Ar_wlehfAgpiWUm8` |
| `default.py` | `minisweagent/agents/default.py` | `6O-Ko2WULXOcLsXLCHn2Dzd9LcLejsZwqu3zuvtFpMI` |
| `LICENSE.md` | `mini_swe_agent-2.4.6.dist-info/licenses/LICENSE.md` | `D3luWPkdHAe7LBsdD4vzqDAXw6Xewb3G-uczss0uh1s` |

## Modifications

Nothing else differs from the wheel:

1. Every `.py` file starts with one added comment line, `# Vendored from mini-swe-agent 2.4.6
   (MIT): minisweagent/<original path>. ...`.
2. The `from minisweagent...` import lines are rewritten to the flat vendored paths:
   - `actions_toolcall.py`: `minisweagent.exceptions` -> `swarmlab.vendor.minisweagent.exceptions`,
     `minisweagent.models.utils.openai_multimodal` -> `swarmlab.vendor.minisweagent.openai_multimodal`.
   - `default.py`: `minisweagent` -> `swarmlab.vendor.minisweagent`, `minisweagent.exceptions` ->
     `swarmlab.vendor.minisweagent.exceptions`, `minisweagent.utils.serialize` ->
     `swarmlab.vendor.minisweagent.serialize`.
3. `__init__.py` is reduced to a new module docstring, `__version__ = "2.4.6"`, the `typing`
   import and the `Model` and `Environment` protocols (verbatim, with their `# === Protocols ===`
   comment). The original's global config directory creation, `.env` loading, startup banner,
   logger, `package_dir` and `Agent` protocol are left out: they are import-time side effects and
   dependencies (python-dotenv, platformdirs, rich) swarmlab does not use.

Formatting is the original's (ruff format would change some of it), and the ruff findings in these
files are ignored in `pyproject.toml` (`[tool.ruff.lint.per-file-ignores]`), not fixed.

## Re-vendoring on a version bump

1. Install the new version somewhere outside the swarmlab environment (`pip install
   mini-swe-agent==X.Y.Z --target <dir>`), and read its changelog for changes to the agent loop,
   exceptions or tool-call helpers.
2. Copy the files in the table over the ones here, add the header line, rewrite the
   `from minisweagent...` imports as above, and reduce `__init__.py` as above (keep the protocols
   verbatim). If `default.py` or `actions_toolcall.py` gained a `minisweagent` import, vendor that
   file too and add it to the table.
3. Copy `licenses/LICENSE.md` and update the RECORD hashes and the version in this file.
4. Update `MSA_VERSION` in `swarmlab/participants/minisweagent.py` (a test checks it equals
   `__version__` here), and re-check the templates copied from `config/benchmarks/programbench.yaml`
   (the cross-check tests run when the real package is installed).
