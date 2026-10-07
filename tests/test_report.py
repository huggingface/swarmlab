"""M4 acceptance 3: `swarmlab report` reproduces the M2 report; `swarmlab prompts` renders the
exact prompts of each participant group without a model call."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from swarmlab import Experiment
from swarmlab.cli import app
from swarmlab.prompts_cmd import render_prompts
from swarmlab.providers.anthropic import AnthropicProvider
from swarmlab.providers.fake import FakeProvider
from swarmlab.providers.openai_compat import OpenAICompatProvider
from swarmlab.report import build_report

from .helpers import Chatter, llm_agent_experiment

REPO = Path(__file__).resolve().parent.parent
M2_RUNS = Path(os.environ.get("AM_LOCAL", "/nonexistent")) / "runs" / "m2"
M2_NOTE = REPO / "docs" / "notes" / "m2-phase1-report-2026-10-06.md"
cli = CliRunner()


@pytest.mark.skipif(not (M2_RUNS / "m2-flaggame__bc-haiku__s1").is_dir(),
                    reason="archived M2 runs not on this machine")
def test_report_reproduces_the_m2_report(tmp_path):
    out = tmp_path / "report.md"
    res = cli.invoke(app, ["report", str(M2_RUNS), "--out", str(out),
                           "--title", "M2 phase-1 Flag Game report"])
    assert res.exit_code == 0, res.output
    assert out.read_text() == M2_NOTE.read_text()


@pytest.fixture(scope="module")
def runs_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("runs")
    Experiment.from_yaml(REPO / "examples" / "flaggame_m1a.yaml", "broadcast").run_all(
        [1, 2], max_rounds=4, out=out)
    Experiment.from_yaml(REPO / "examples" / "flaggame_m1a.yaml", "gossip").run_all(
        [1], max_rounds=6, out=out)
    (out / "notarun").mkdir()
    return out


def test_report_sections_and_round_span(runs_dir):
    text = build_report(runs_dir, "demo")
    assert text.startswith("# demo\n")
    for section in ("## Summary", "## Trajectories (mean over seeds)",
                    "## Where the swarm went (final committed guesses)", "## Probe vs world belief",
                    "## Reading behaviour", "## Tool protocol health"):
        assert section in text
    assert "| broadcast | 2 |" in text and "| gossip | 1 |" in text
    assert "read_rate r2-6" in text and "| r6 |" in text and "| r7 |" not in text
    assert "flaggame-m1a__broadcast__s2" in text


def test_report_cli_json(runs_dir):
    res = cli.invoke(app, ["report", str(runs_dir), "--json"])
    assert res.exit_code == 0, res.output
    assert json.loads(res.output)["report"].startswith("# swarmlab report")
    assert cli.invoke(app, ["report", str(runs_dir / "missing")]).exit_code == 2


@pytest.fixture
def no_model_calls(monkeypatch):
    async def boom(self, request):  # pragma: no cover - must never run
        raise AssertionError("prompts must not call a model")

    for cls in (FakeProvider, AnthropicProvider, OpenAICompatProvider):
        monkeypatch.setattr(cls, "complete", boom)


def test_prompts_match_the_first_request_of_a_run(tmp_path, no_model_calls, monkeypatch):
    exp = llm_agent_experiment(3, name="prompts", agent_kw={"tool_protocol": "json"})
    exp.participants += [Chatter(), Chatter()]
    rows = render_prompts(exp, seed=5)
    assert [(r["type"], r["count"], r["agent"]) for r in rows] == [
        ("llm", 3, "a000"), ("tests.helpers:Chatter", 2, "a003")]
    assert rows[1]["system"] is None and "not prompted" in rows[1]["note"]
    assert "Tool protocol:" in rows[0]["system"] and rows[0]["user"].startswith("Round 1.")
    # the same strings reach the provider in a real run
    monkeypatch.undo()
    run = exp.run(seed=5, max_rounds=1, out=tmp_path / "runs")
    first = next(e for e in run.events_all if e.type == "inference_attempt" and e.agent == "a000")
    h = first.request_hash
    req = json.loads((run.dir / "blobs" / h[:2] / h).read_text())
    assert req["messages"][0]["content"] == rows[0]["system"]
    assert req["messages"][1]["content"] == rows[0]["user_parts"]


def test_prompts_cli_on_every_example_arm(no_model_calls):
    spec = REPO / "examples" / "flaggame_m1a.yaml"
    res = cli.invoke(app, ["prompts", str(spec), "--arm", "llm"])
    assert res.exit_code == 0, res.output
    assert "--- system prompt ---" in res.output and "--- round 1 user message ---" in res.output
    assert "You are agent a000" in res.output and "Round 1." in res.output
    res = cli.invoke(app, ["prompts", str(spec), "--arm", "gossip", "--json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["seed"] == 1 and data["groups"][0]["system"] is None
    assert cli.invoke(app, ["prompts", str(spec)]).exit_code == 2  # several arms, no --arm


def test_skill_points_only_at_commands_and_flags_that_exist():
    """skill/SKILL.md rule: every `swarmlab ...` it mentions is a real command with real flags."""
    import re

    text = (REPO / "skill" / "SKILL.md").read_text()
    uses = re.findall(r"(?:`|^)swarmlab ((?:job )?[a-z][a-z-]*)([^`\n]*)", text, re.MULTILINE)
    assert len(uses) > 20
    helps: dict[str, str] = {}
    for cmd, rest in uses:
        if cmd not in helps:
            res = cli.invoke(app, [*cmd.split(), "--help"], terminal_width=200)
            assert res.exit_code == 0, f"swarmlab {cmd} is not a command"
            helps[cmd] = res.output
        for flag in re.findall(r"(?<![\w-])--[a-z][a-z-]*", rest.split("`")[0]):
            assert flag in helps[cmd], f"swarmlab {cmd} has no {flag}"
    for cmd in ("doctor", "prompts", "estimate", "report", "publish", "fetch-published",
                "job run", "job fetch", "resume", "fork", "replay", "export", "view"):
        assert f"swarmlab {cmd}" in text, cmd


def test_report_sets_simulated_runs_apart_and_out_is_quiet(tmp_path):
    out = tmp_path / "runs"
    spec = REPO / "examples" / "flaggame_m1a.yaml"
    Experiment.from_yaml(spec, "broadcast").run_all([1], max_rounds=2, out=out)
    Experiment.from_yaml(spec, "llm").run_all([1], max_rounds=2, out=out)  # fake:reader
    text = build_report(out)
    assert "Simulated runs (only `fake:` models" in text and "flaggame-m1a__llm__s1" in text
    assert "| llm" not in text and "| broadcast | 1 |" in text
    assert "Total spend (ledger, swarm + measurement) over the 1 real run(s): $0.000" in text
    assert "1 simulated run(s)" in text
    full = build_report(out, include_fake=True)
    assert "| llm (simulated) | 1 |" in full and "| broadcast | 1 |" in full
    assert "over the 1 real run(s)" in full  # simulated spend is never counted
    md = tmp_path / "report.md"
    res = cli.invoke(app, ["report", str(out), "--out", str(md)])
    assert res.exit_code == 0 and res.output.strip().splitlines() == [res.output.strip()]
    assert res.output.startswith(f"wrote {md}") and md.read_text() == text + "\n"
    res = cli.invoke(app, ["report", str(out), "--out", str(md), "--stdout", "--include-fake"])
    assert res.exit_code == 0 and "| llm (simulated) | 1 |" in res.output
