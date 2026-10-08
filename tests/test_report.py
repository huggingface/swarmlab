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
    assert since_m2(out.read_text()) == M2_NOTE.read_text()


def since_m2(text: str) -> str:
    """The report without the rows and sections added after the M2 note was written."""
    out, drop_blank, health, traj, dropped = [], False, False, False, False
    for line in text.splitlines(keepends=True):
        if line.startswith("## "):  # sections added later: protocol health, coloring
            dropped = line.startswith(("## Protocol health", "## Coloring", "## Terminal states"))
            traj = line.startswith("## Trajectories")
        if dropped:
            continue
        if traj and line.startswith("| ") and "." in line.split(" | ")[0]:
            continue  # every other logged metric (full dotted name), added later
        health = health or line.startswith("## Tool protocol health")
        if health and line.startswith("|---"):  # max_tokens, rejected tool calls appended later
            line = line.replace("---|", "", 2)
        elif health and line.startswith("|"):
            line = line.rstrip("\n").rstrip(" |").rsplit(" | ", 2)[0] + " |\n"
        if line.startswith(("| post_rate |", "Probes skipped:")):
            drop_blank = line.startswith("Probes skipped:")
            continue
        if drop_blank and line == "\n":
            drop_blank = False
            continue
        out.append(line)
    return "".join(out)


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
    assert "| post_rate |" in text


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


def test_skill_checklist_scales_to_the_budget():
    text = (REPO / "skill" / "SKILL.md").read_text()
    short = text.split("**Total under $2: the short checklist.**")[1].split("**Total of $2 or more")[0]
    assert "N <= 6" in short and "50% of the total" in short and "`soft_usd: 0`" in short
    # the second arm's caps come from the first arm's measured spend, caps live per arm
    assert "spend_usd" in short and "arms.A.budget" in short and "existing:" in short
    assert "swarmlab prompts SPEC.yaml --arm A" in short and "diff A.txt B.txt" in short
    assert "second agent" not in short.replace("without a second agent", "")
    assert "total_usd" in text and "all eleven items" in text


def test_tool_health_counts_length_and_max_tokens_finishes(runs_dir):
    from types import SimpleNamespace

    from swarmlab.report import scan

    def resp(fr):
        return SimpleNamespace(type="inference_response", round=1, agent="a000", cached=False,
                               finish_reason=fr)

    d = scan(SimpleNamespace(events_all=[resp("length"), resp("max_tokens"), resp("max_tokens"),
                                         resp("tool_use")]))
    assert (d["length"], d["max_tokens"], d["nresp"]) == (1, 2, 4)
    text = build_report(runs_dir, "demo")
    assert "| length (responses) | errored turns | cache hits/responses | max_tokens (responses) |" in text


def test_init_starter_describes_gossip_as_one_partner(tmp_path):
    res = cli.invoke(app, ["init", "demo", "--dir", str(tmp_path)])
    assert res.exit_code == 0, res.output
    text = (tmp_path / "demo.yaml").read_text()
    assert "one random partner per round by default (k=1)" in text
    assert "a few neighbours" not in text


def test_protocol_health_and_metrics_for_a_custom_world(tmp_path):
    """Any world gets the protocol-health section and every logged metric; world-specific
    sections appear only for their world type."""
    from .helpers import site_experiment

    out = tmp_path / "runs"
    site_experiment().run(seed=1, max_rounds=3, out=out)
    text = build_report(out, include_fake=True)
    for section in ("## Trajectories (mean over seeds)", "## Tool protocol health",
                    "## Protocol health"):
        assert section in text
    for section in ("## Where the swarm went", "## Probe vs world belief", "## Coloring"):
        assert section not in text
    traj = text.split("## Trajectories")[1].split("## ")[0]
    assert "| sites.max_share | " in traj and "| accuracy |" not in traj  # not a fixed list
    health = text.split("## Protocol health")[1]
    assert "| turn end kinds | end_turn 9, error 3 |" in health
    assert ("| turns errored | 3 of 12; first: `sites__s1` r1 a003: RuntimeError: provider said "
            "no in round 1 |") in health
    assert "| finish reasons (responses) | tool_use 18 |" in health
    assert "| model calls per turn, swarm (mean / max) | 1.50 / 2 |" in health
    assert "| latency s, uncached (median / p90 / max) | 0.00 / 0.00 / 0.00 |" in health
    assert "| retries (responses retried / extra attempts) | 0 / 0 |" in health
    assert "| tool calls rejected / answered | 0 / 18 |" in health
    assert "world actions not accepted / committed | 1 / 9 (unknown_site 1) |" in health
    assert "| probes skipped | no probes |" in health
    assert "| cost per round, swarm + measurement (mean / max) | $0." in health

    # a coloring run in the same dir adds the coloring section, still no Flag Game sections
    Experiment.from_yaml(REPO / "examples" / "coloring_s0.yaml", "row_major").run_all(
        [1], max_rounds=2, out=out)
    text = build_report(out, include_fake=True)
    assert "## Coloring (final grid)" in text and "coloring-s0__row_major__s1" in text
    assert "| coloring.coverage |" in text and "## Where the swarm went" not in text


def test_first_error_line_and_percentile():
    from swarmlab.report import first_error_line, pct

    tb = ("Traceback (most recent call last):\n  File \"x.py\", line 1, in f\n    boom()\n"
          "ValueError: bad thing\nmore detail\n")
    assert first_error_line(tb) == "ValueError: bad thing"
    assert first_error_line("HTTP 400: chat_template_kwargs\nbody") == "HTTP 400: chat_template_kwargs"
    assert first_error_line(None) == "(no error text)"
    assert pct([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 90) == 9 and pct([], 90) is None


def test_readme_truth_metric_example_runs(tmp_path):
    """The analysis guide's `verify()`-based metric sample works as written (field notes item 10)."""
    import re

    from .helpers import site_experiment

    readme = (REPO / "docs" / "guide" / "analysis.md").read_text()
    section = readme.split("### Analysis patterns")[1].split("\n## ")[0]
    code = re.search(r"```python\n(.*?)```", section, re.DOTALL).group(1)
    assert len(code.strip().splitlines()) <= 17
    ns: dict = {}
    exec(compile(code, "analysis.md", "exec"), ns)  # noqa: S102 - our own docs sample
    exp = site_experiment(crash=False)
    exp.metrics.append(ns["BestSiteShare"]())
    run = exp.run(seed=2, max_rounds=3, out=tmp_path)
    series = run.metrics["sites.best_share"]
    assert [r for r, _, _ in series] == [1, 2, 3]
    picks = [e for e in run.events if e["type"] == "action_committed" and e["accepted"]]
    hits = sum(e["feedback"]["site"] == "A" for e in picks)
    assert series[-1][1:] == (hits / len(picks), len(picks))
