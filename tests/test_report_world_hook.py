"""`swarmlab report` for any world: the generic "Final score" table (every scalar key of the
recorded `score()`) and the `World.report_sections` hook, which degrades to a note when the
world cannot be built."""
from __future__ import annotations

import json

from swarmlab.experiment import Run
from swarmlab.report import build_report

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


def test_failing_or_unimportable_world_degrades_to_a_note(tmp_path, monkeypatch):
    out = tmp_path / "broken"
    run_with(BrokenHook(), out)
    text = build_report(out, include_fake=True)
    assert "## Final score" in text
    assert "(world report sections unavailable: RuntimeError: hook exploded)" in text

    # a world type that does not import here: the runs still report, the hook is a note
    out = tmp_path / "gone"
    (run,) = run_with(ScoredSites(), out)
    meta = json.loads((run.dir / "run.json").read_text())
    meta["spec"]["world"]["type"] = "swarmlab_no_such_module:GoneWorld"
    (run.dir / "run.json").write_text(json.dumps(meta))
    monkeypatch.setattr(Run, "load", classmethod(lambda cls, d, run_id=None: cls(d)))  # no replay
    text = build_report(out, include_fake=True)
    assert "## Final score" in text and f"| {run.id} | 2 | " in text
    note = next(ln for ln in text.splitlines() if ln.startswith("(world report sections unavailable"))
    assert "swarmlab_no_such_module" in note
