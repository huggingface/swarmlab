"""Built-in vLLM serve profiles: per-model `vllm serve` flags for `swarmlab job run`.

A profile holds only what the server needs to load the model and parse its output: dtype,
`--language-model-only` for multimodal checkpoints served as text, the tool-call parser and the
reasoning parser. It also holds the request extra the launcher's readiness probe sends (e.g.
`chat_template_kwargs: {enable_thinking: false}` for templates that think by default). It never
holds sampling params (temperature, top_p, max_tokens): those belong to the spec's participant
`params` / `extra`, so the spec stays the single record of how the model was called.

Resolution (`resolve_flags`): an explicit `--vllm-flags` string wins, then the profile (`--profile
auto` looks up `--model`; `--profile <model id>` forces one; `--profile none` skips the table),
then `DEFAULT_VLLM_FLAGS` (the Qwen3.5 textworld flags), so a model not in the table is served
exactly as before. `--vllm-flags-append` is added to whichever of the three won.

The launcher always adds `--served-model-name <model> --host 127.0.0.1 --port <port>
--max-model-len <n> --gpu-memory-utilization <f>` (and `--revision` when given) itself; flag
strings must not repeat them (`check_flags` rejects that), since the job spec's provider URL and
the readiness check depend on them.

Every profile below was validated under vLLM 0.27.1, BF16, TP=1, on an A10G (24 GB), except
Qwen/Qwen3.5-9B (A100). The launcher's default vLLM is newer (0.30.0); `describe` notes the
mismatch and `--vllm-version 0.27.1` reproduces the validated setup. The parser names used here
(`qwen3`, `qwen3_coder`, `hermes`, `lfm2`, `minimax_m2`) are registered in both 0.27.1 and 0.30.0.
"""
from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Any

from ..spec import SpecError

DEFAULT_VLLM_FLAGS = ("--dtype bfloat16 --language-model-only --reasoning-parser qwen3 "
                      "--enable-auto-tool-choice --tool-call-parser qwen3_coder")
# the request extra the readiness probe sends when no profile applies (unchanged behaviour)
DEFAULT_PROBE_EXTRA: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}
NO_THINKING: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}
# flags `bootstrap.sh` passes itself; a flag string repeating one is rejected
LAUNCHER_FLAGS = ("--served-model-name", "--host", "--port", "--max-model-len",
                  "--gpu-memory-utilization", "--revision")
PROFILE_CHOICES_HELP = "auto (look up --model), none (the Qwen default), or a model id in the table"


@dataclass(frozen=True)
class ServeProfile:
    """How to serve one model with vLLM (flags + readiness-probe request extra)."""

    model: str
    flags: str
    probe_extra: dict[str, Any] = field(default_factory=dict)
    validated: str = "vLLM 0.27.1, BF16, TP=1, a10g"
    validated_vllm: str = "0.27.1"
    note: str = ""


PROFILES: dict[str, ServeProfile] = {p.model: p for p in (
    ServeProfile(
        "Qwen/Qwen3.5-2B",
        "--dtype bfloat16 --language-model-only --enable-auto-tool-choice "
        "--tool-call-parser qwen3_coder",
        NO_THINKING, note="multimodal checkpoint served text-only; non-thinking per request"),
    ServeProfile(
        "Qwen/Qwen3.5-9B",
        "--dtype bfloat16 --language-model-only --enable-auto-tool-choice "
        "--tool-call-parser qwen3_coder",
        NO_THINKING, validated="vLLM 0.27.1, BF16, TP=1, a100",
        note="multimodal checkpoint served text-only; non-thinking per request"),
    ServeProfile(
        "ibm-granite/granite-4.1-3b",
        "--dtype bfloat16 --enable-auto-tool-choice --tool-call-parser hermes",
        note="text-only; no reasoning parser"),
    ServeProfile(
        "Qwen/Qwen3-4B-Instruct-2507",
        "--dtype bfloat16 --enable-auto-tool-choice --tool-call-parser hermes",
        note="text-only instruct model (no thinking mode); no reasoning parser"),
    ServeProfile(
        "ibm-granite/granite-4.2-8b",
        "--dtype bfloat16 --enable-auto-tool-choice --tool-call-parser qwen3_coder",
        NO_THINKING, note="non-thinking per request; no reasoning parser"),
    ServeProfile(
        "LiquidAI/LFM2.5-2.6B",
        "--dtype bfloat16 --enable-auto-tool-choice --tool-call-parser lfm2 "
        "--reasoning-parser minimax_m2",
        note="text-only; minimax_m2 splits <think> output into reasoning"),
)}


def find_profile(model: str) -> ServeProfile | None:
    """The profile for a model id (exact match, then case-insensitive), or None."""
    if model in PROFILES:
        return PROFILES[model]
    low = model.lower()
    return next((p for k, p in PROFILES.items() if k.lower() == low), None)


def select_profile(model: str, profile: str | None = "auto") -> ServeProfile | None:
    """`auto` (or None) -> the table entry for `model` if any; `none` -> no profile; otherwise
    the named profile, which must exist."""
    name = (profile or "auto").strip()
    if name.lower() == "auto":
        return find_profile(model)
    if name.lower() == "none":
        return None
    p = find_profile(name)
    if p is None:
        raise SpecError(f"--profile {name!r}: no such profile; use {PROFILE_CHOICES_HELP}: "
                        + ", ".join(PROFILES))
    return p


def check_flags(flags: str, what: str = "--vllm-flags") -> list[str]:
    """Tokenise a flag string; reject flags the launcher passes itself. Returns the tokens."""
    try:
        tokens = shlex.split(flags)
    except ValueError as e:
        raise SpecError(f"{what}: cannot parse {flags!r}: {e}") from e
    clash = []
    for t in tokens:
        if t.startswith("--"):
            name = t.split("=", 1)[0].replace("_", "-")
            if name in LAUNCHER_FLAGS:
                clash.append(name)
    if clash:
        raise SpecError(f"{what} must not contain {sorted(set(clash))}: the launcher always passes "
                        f"{' '.join(LAUNCHER_FLAGS)} itself (use --max-model-len / --revision)")
    return tokens


@dataclass(frozen=True)
class ResolvedFlags:
    flags: str
    source: str           # "--vllm-flags" | "profile" | "default"
    profile: str | None   # the profile's model id when one was selected
    probe_extra: dict[str, Any]
    validated_vllm: str | None


def resolve_flags(model: str, *, vllm_flags: str | None = None, profile: str | None = "auto",
                  append: str | None = None) -> ResolvedFlags:
    """Explicit flags > profile > default; `append` goes after whichever won.

    The probe extra follows the selected profile even when `vllm_flags` replaces its flags (it
    describes the model's chat template, not the server); without a profile it is the default.
    """
    prof = select_profile(model, profile)
    if vllm_flags is not None:
        flags, source = vllm_flags, "--vllm-flags"
    elif prof is not None:
        flags, source = prof.flags, "profile"
    else:
        flags, source = DEFAULT_VLLM_FLAGS, "default"
    if append:
        flags = f"{flags} {append}".strip()
    flags = shlex.join(check_flags(flags))
    return ResolvedFlags(
        flags=flags, source=source, profile=prof.model if prof else None,
        probe_extra=dict(prof.probe_extra) if prof else dict(DEFAULT_PROBE_EXTRA),
        validated_vllm=prof.validated_vllm if prof else None)
