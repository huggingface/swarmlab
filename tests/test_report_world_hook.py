"""`swarmlab report` for any world: the generic "Final score" table (every scalar key of the
recorded `score()`, Markdown-escaped), the `World.report_sections` hook (once per world type,
with that type's runs only), which degrades to a note when the world cannot be built, and runs
whose world does not import here, read from run.json without replay."""
from __future__ import annotations

import importlib
import re
import sys
import textwrap
from typing import ClassVar

from swarmlab.report import build_report, table

from .helpers import SiteWorld, site_experiment


class ScoredSites(SiteWorld):
    """SiteWorld with a mixed score and its own report section."""

    def score(self):
        picks = dict(self.picks)
        total = sum(picks.values())
        return {"total_picks": total, "share_a": picks["A"] / total if total else None,
                "leader": max(picks, key=picks.get), "unanimous": len({v for v in picks.values() if v}) <= 1,
                "missing": None, "picks": picks, "by_round": [total]}

    def report_sections(self, arms):
        lines = ["## Sites by arm", "", "| arm | runs | mean total picks |", "|---|---|---|"]
        for arm, runs in arms.items():
            totals = [r.score["total_picks"] for r in runs]
            lines.append(f"| {arm} | {len(runs)} | {sum(totals) / len(totals):.1f} |")
        return [*lines, ""]


class BrokenHook(SiteWorld):
    def report_sections(self, arms):
        raise RuntimeError("hook exploded")


def run_with(world, out, seeds=(1,), arm=None):
    exp = site_experiment(crash=False)
    exp.world = world
    if arm:
        exp.arm = arm
    return [exp.run(seed=s, max_rounds=2, out=out) for s in seeds]


def test_custom_world_scalar_score_keys_in_final_score(tmp_path):
    runs = run_with(ScoredSites(), tmp_path, seeds=(1, 2))
    text = build_report(tmp_path, include_fake=True)
    section = text.split("## Final score")[1].split("\n## ")[0]
    head = next(ln for ln in section.splitlines() if ln.startswith("| run "))
    # run.json stores score() with sorted keys, so first-seen order is alphabetical here
    assert head == "| run | rounds | leader | missing | share_a | total_picks | unanimous |"
    row = next(ln for ln in section.splitlines() if ln.startswith(f"| {runs[0].id} |"))
    sc = runs[0].score
    assert row == (f"| {runs[0].id} | 2 | {sc['leader']} | n/a | {sc['share_a']:.3f} | "
                   f"{sc['total_picks']} | {sc['unanimous']} |")
    assert "| picks |" not in head and "by_round" not in head  # nested values are skipped
    for name in ("## Where the swarm went", "## Terminal states"):  # no guesses, no truth
        assert name not in text


def test_world_report_sections_follow_the_score_table(tmp_path):
    runs = run_with(ScoredSites(), tmp_path, seeds=(1, 2))
    text = build_report(tmp_path, include_fake=True)
    assert text.index("## Final score") < text.index("## Sites by arm")
    mean_total = sum(r.score["total_picks"] for r in runs) / 2
    rows = [ln for ln in text.split("## Sites by arm")[1].splitlines() if ln.startswith("| ")]
    assert rows == ["| arm | runs | mean total picks |", f"| ? (simulated) | 2 | {mean_total:.1f} |"]
    assert "world report sections unavailable" not in text


def test_failing_hook_degrades_to_a_note(tmp_path):
    run_with(BrokenHook(), tmp_path)
    text = build_report(tmp_path, include_fake=True)
    assert "## Final score" in text
    assert "(world report sections unavailable: RuntimeError: hook exploded)" in text


GONE = textwrap.dedent("""
    from tests.helpers import SiteWorld


    class GoneWorld(SiteWorld):
        def score(self):
            return {"total_picks": sum(self.picks.values()), "leader": "A"}

        def report_sections(self, arms):
            return ["## Gone world section", ""]
""")


def test_unimportable_world_falls_back_to_the_recorded_run(tmp_path, monkeypatch):
    """The real loading path: `Run.load` cannot replay a run whose world module is gone, so the
    report reads it from run.json (score, rounds, metrics, events) and names it in the header."""
    mods = tmp_path / "mods"
    mods.mkdir()
    name = "swarmlab_report_gone_world"
    (mods / f"{name}.py").write_text(GONE)
    monkeypatch.syspath_prepend(str(mods))
    world = importlib.import_module(name).GoneWorld()
    out = tmp_path / "runs"
    (run,) = run_with(world, out)
    assert run.spec.world.type == f"{name}:GoneWorld"
    # make the module unimportable: off sys.path, out of sys.modules, file gone
    monkeypatch.delitem(sys.modules, name)
    sys.path.remove(str(mods))
    (mods / f"{name}.py").unlink()
    importlib.invalidate_caches()
    # a run that is not ended is still skipped as before
    (out / "not_ended").mkdir()
    (out / "not_ended" / "run.json").write_text('{"status": "running"}')

    text = build_report(out, include_fake=True)
    head = text.split("## Summary")[0]
    assert "skipped (not ended / unreadable): not_ended" in head
    warn = next(ln for ln in head.splitlines() if ln.startswith("reconstruction unavailable"))
    assert warn.startswith(f"reconstruction unavailable for {run.id}: ModuleNotFoundError: ")
    assert name in warn
    final = text.split("## Final score")[1]
    assert "| run | rounds | leader | total_picks |" in final
    assert f"| {run.id} | 2 | A | {run.score['total_picks']} |" in final
    assert "| sites.max_share | " in text.split("## Trajectories")[1]  # metrics from the log
    assert "## Where the swarm went" not in text  # no rebuilt world, no truth
    assert "## Gone world section" not in text
    note = next(ln for ln in text.splitlines() if ln.startswith("(world report sections unavailable"))
    assert name in note


class Weird(SiteWorld):
    def score(self):
        return {"left|right\nnext line": "a|b\r\nc", "plain": 1}


def cells(line):
    return re.split(r"(?<!\\)\|", line)[1:-1]


def test_table_escapes_pipes_and_line_breaks(tmp_path):
    assert table(["a|b", "c\nd"], [["x\r\ny", "p|q"]]) == [
        "| a\\|b | c d |", "|---|---|", "| x y | p\\|q |"]
    (run,) = run_with(Weird(), tmp_path)
    text = build_report(tmp_path, include_fake=True)
    section = text.split("## Final score")[1].split("\n## ")[0]
    rows = [ln for ln in section.splitlines() if ln.startswith("|")]
    assert len(rows) == 3  # header, rule, one run: no line was split
    head, rule, row = rows
    assert [c.strip() for c in cells(head)] == ["run", "rounds", "left\\|right next line", "plain"]
    assert len(cells(rule)) == len(cells(row)) == 4
    assert [c.strip() for c in cells(row)] == [run.id, "2", "a\\|b c", "1"]


class Recording(ScoredSites):
    """ScoredSites that records the arms its hook receives (on the class: the report builds
    its own instance)."""

    seen: ClassVar[list] = []

    def report_sections(self, arms):
        Recording.seen.append({arm: [r.id for r in runs] for arm, runs in arms.items()})
        return super().report_sections(arms)


def test_hooks_run_per_world_type_with_their_own_runs(tmp_path):
    """`a_plain` (SiteWorld, no hook) sorts before `z_custom`; the custom world's section still
    appears, under its own heading, and its hook sees only its own runs."""
    Recording.seen = []
    plain = run_with(SiteWorld(), tmp_path, arm="a_plain")
    custom = run_with(Recording(), tmp_path, seeds=(1, 2), arm="z_custom")
    text = build_report(tmp_path, include_fake=True)
    assert Recording.seen == [{"z_custom (simulated)": [r.id for r in custom]}]
    wtype = Recording.type_name()
    assert f"### World {wtype}" in text
    assert text.index(f"### World {wtype}") < text.index("## Sites by arm")
    assert f"### World {SiteWorld.type_name()}" not in text  # no lines, no heading
    hook = text.split("## Sites by arm")[1]
    assert "| z_custom (simulated) | 2 |" in hook and "a_plain" not in hook
    final = text.split("## Final score")[1]
    assert f"| {plain[0].id} | 2 |" in final and f"| {custom[0].id} | 2 |" in final
    assert "world report sections unavailable" not in text

    # one world type: no heading
    Recording.seen = []
    one = tmp_path / "one"
    run_with(Recording(), one, arm="z_custom")
    text = build_report(one, include_fake=True)
    assert "### World" not in text and "## Sites by arm" in text and len(Recording.seen) == 1
