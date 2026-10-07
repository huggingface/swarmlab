"""Experiment-wide total cap (`budget.total_usd`), planned caps and the soft/hard gap warning."""
import pytest
import yaml

from swarmlab import Experiment
from swarmlab.budget import TotalBudgetWarning, total_cap_refusal
from swarmlab.spec import SpecError, load_experiment_yaml

from .test_cli_run_all import invoke

CAP = """\
name: cap
seeds: [1, 2, 3]
budget: {soft_usd: 0.04, hard_usd: 0.05, total_usd: 0.12}
providers:  # `paid:` is a FakeProvider standing in for a billed provider (fake: arms are exempt)
  paid: {type: fake, params: {pricing: {"*": [10.0, 50.0, 1.0]}}}
options: {max_rounds: 2}
arms:
  A:
    world: {type: flaggame}
    participants: [{type: llm, count: 2, params: {model: "paid:reader", max_tokens: 64, max_calls: 2}}]
    metrics: [belief.consensus]
"""


def spec_file(tmp_path, text=CAP, **budget):
    doc = yaml.safe_load(text)
    doc["budget"].update(budget)
    p = tmp_path / "cap.yaml"
    p.write_text(yaml.safe_dump(doc))
    return p


def test_refusal_rule():
    assert total_cap_refusal(0.5, 0.5, 0) is None  # no total cap
    assert total_cap_refusal(0.5, 0.5, 1.0) is None  # exactly fits
    assert "> total_usd" in total_cap_refusal(0.51, 0.5, 1.0)
    assert "hard_usd" in total_cap_refusal(0.0, 0.0, 1.0)  # an unbounded run cannot be admitted


def test_run_stops_before_a_run_that_could_break_the_total(tmp_path):
    spec = spec_file(tmp_path)
    out = tmp_path / "runs"
    res, data = invoke("run", spec, "--out", out, "--yes", "--json")
    assert res.exit_code == 1, res.output
    outcomes = [(r["run_id"], r["outcome"]) for r in data["runs"]]
    assert outcomes == [("cap__A__s1", "ran"), ("cap__A__s2", "ran"), ("cap__A__s3", "capped")]
    assert data["capped"]["skipped"] == ["cap__A__s3"] and data["capped"]["total_usd"] == 0.12
    spent = sum(r["spend"]["swarm"] + r["spend"]["measurement"] for r in data["runs"][:2])
    assert spent + 0.05 > 0.12 and data["capped"]["spent"] == pytest.approx(spent)
    assert not (out / "cap__A__s3").exists()
    # planned caps are printed up front (stderr under --json), the skip is reported
    assert "caps: arm=A per run soft=$0.04 hard=$0.05" in res.stderr
    assert "caps: total $0.12 for the 3 run(s)" in res.stderr
    assert "skipped 1 run(s): cap__A__s3" in res.stderr
    # raising the total cap does not change spec hashes: finished runs are skipped, s3 runs
    res, data = invoke("run", spec_file(tmp_path, total_usd=1.0), "--out", out, "--yes", "--json")
    assert res.exit_code == 0, res.output
    assert [r["outcome"] for r in data["runs"]] == ["skipped", "skipped", "ran"]


def test_total_cap_needs_hard_ceilings_and_lives_at_top_level(tmp_path):
    res, _ = invoke("run", spec_file(tmp_path, hard_usd=0), "--out", tmp_path / "r", "--yes")
    assert res.exit_code == 2 and "needs hard_usd > 0" in res.stderr
    doc = yaml.safe_load(CAP)
    doc["arms"]["A"]["budget"] = {"total_usd": 1.0}
    (tmp_path / "arm.yaml").write_text(yaml.safe_dump(doc))
    with pytest.raises(SpecError, match="top-level budget"):
        load_experiment_yaml(tmp_path / "arm.yaml")


def test_total_usd_is_not_part_of_the_spec_hash_or_documents(tmp_path):
    a = Experiment.from_yaml(spec_file(tmp_path, total_usd=0), "A")
    assert "total_usd" not in a.budget.model_dump()  # absent when off: old documents unchanged
    b = Experiment.from_yaml(spec_file(tmp_path, total_usd=5.0), "A")
    assert b.budget.total_usd == 5.0
    assert a.spec_hash(1) == b.spec_hash(1)


def test_python_run_all_honours_the_total(tmp_path):
    exp = Experiment.from_yaml(spec_file(tmp_path), "A")
    with pytest.warns(TotalBudgetWarning, match=r"skipped seeds \[3\]"):
        runs = exp.run_all([1, 2, 3], out=tmp_path / "runs")
    assert [r.id for r in runs] == ["cap__A__s1", "cap__A__s2"]


def test_soft_hard_gap_warning(tmp_path):
    res, _ = invoke("run", spec_file(tmp_path, total_usd=0), "--seed", 1, "--out", tmp_path / "r",
                    "--yes")
    assert res.exit_code == 0, res.output
    assert ("warning: arm=A: hard_usd - soft_usd = $0.0100 is less than half of one round's "
            "estimated worst case") in res.stdout
    assert "caps: no total cap" in res.stdout
    res, _ = invoke("run", spec_file(tmp_path, total_usd=0, soft_usd=0.04, hard_usd=1.0), "--seed", 1,
                    "--out", tmp_path / "r2", "--yes")
    assert res.exit_code == 0 and "warning:" not in res.stdout


RESUMED = CAP.replace("seeds: [1, 2, 3]", "seeds: [1, 2]").replace(
    "budget: {soft_usd: 0.04, hard_usd: 0.05, total_usd: 0.12}",
    "budget: {soft_usd: 0.02, hard_usd: 0.05, total_usd: 0.1}").replace(
    "options: {max_rounds: 2}", "options: {max_rounds: 4}")


def test_total_counts_spend_added_by_a_manual_resume(tmp_path):
    """A run resumed by hand is skipped by `run`; its ledger (resumed spend included) counts."""
    spec = tmp_path / "cap.yaml"
    spec.write_text(RESUMED)
    out = tmp_path / "runs"
    res, data = invoke("run", spec, "--out", out, "--seed", 1, "--arm", "A", "--yes", "--json")
    assert res.exit_code == 0, res.output
    before = data["spend_usd"]
    assert data["end_reason"] == "soft_budget" and before + 0.05 <= 0.1  # s2 would still fit
    res, data = invoke("resume", out / "cap__A__s1", "--add-budget", 0.05, "--json")
    assert res.exit_code == 0, res.output
    after = data["spend_usd"]
    assert after > before and after + 0.05 > 0.1  # now it does not
    assert "resume does not check the experiment's total cap" in res.stderr
    res, data = invoke("run", spec, "--out", out, "--yes", "--json")
    assert res.exit_code == 1, res.output
    assert [r["outcome"] for r in data["runs"]] == ["skipped", "capped"]
    assert data["capped"]["spent"] == pytest.approx(after)
    # the caps lines show the existing run's actual spend, not the spec caps
    assert (f"existing: arm=A seed=1 spent ${after:.4f} actual over "
            f"{data['runs'][0]['last_round']} round(s)") in res.stderr
    assert f"existing runs already spent ${after:.4f} (resumed spend included)" in res.stderr
