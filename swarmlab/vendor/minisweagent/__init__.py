# Vendored from mini-swe-agent 2.4.6 (MIT): minisweagent/__init__.py. Reduced to the version and two protocols; see README.md.
"""A reduced copy of mini-swe-agent 2.4.6's `minisweagent/__init__.py`.

Only `__version__` and the `Model` and `Environment` protocols (which `default.py` imports) are
kept, verbatim. The original's global config directory creation, `.env` loading, startup banner,
logger and `Agent` protocol are left out: they are import-time side effects and dependencies
(python-dotenv, platformdirs, rich) swarmlab does not want.
"""

__version__ = "2.4.6"

from typing import Any, Protocol

# === Protocols ===
# You can ignore them unless you want static type checking.


class Model(Protocol):
    """Protocol for language models."""

    config: Any

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict: ...

    def format_message(self, **kwargs) -> dict: ...

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]: ...

    def get_template_vars(self, **kwargs) -> dict[str, Any]: ...

    def serialize(self) -> dict: ...


class Environment(Protocol):
    """Protocol for execution environments."""

    config: Any

    def execute(self, action: dict, cwd: str = "") -> dict[str, Any]: ...

    def get_template_vars(self, **kwargs) -> dict[str, Any]: ...

    def serialize(self) -> dict: ...
