"""`swarmlab run SPEC` over every arm x seed, and the Python API it uses (run_all, summary, load)."""
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from swarmlab import Experiment, Run
from swarmlab.cli import app

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "flaggame_m1a.yaml"
ALL = [f"flaggame-m1a__{arm}__s{seed}" for arm in ("broadcast", "gossip", "llm") for seed in (1, 2)]
runner = CliRunner()


def invoke(*args, input=None):
    res = runner.invoke(app, [str(a) for a in args], input=input)
    data = json.loads(res.stdout) if "--json" in args and res.stdout.strip() else None
    return res, data


def test_run_all_arms_and_seeds(tmp_path):
    out = tmp_path / "runs"
    # the llm arm has a budget: without --yes (and nothing typed) nothing runs
    res, _ = invoke("run", EXAMPLE, "--max-rounds", 2, "--out", out)
    assert res.exit_code == 1 and "--yes" in res.stderr and not out.exists()
    # answering the question runs everything
    res, _ = invoke("run", EXAMPLE, "--max-rounds", 2, "--out", out, "--arm", "llm", "--seed", 1,
                    input="y\n")
    assert res.exit_code == 0, res.output

    res, data = invoke("run", EXAMPLE, "--max-rounds", 2, "--out", out, "--yes", "--json")
    assert res.exit_code == 0, res.output
    assert [r["run_id"] for r in data["runs"]] == ALL
    outcomes = {r["run_id"]: r["outcome"] for r in data["runs"]}
    assert outcomes.pop("flaggame-m1a__llm__s1") == "skipped"   # ran above with the same spec
    assert set(outcomes.values()) == {"ran"}
    assert all(r["end_reason"] == "max_rounds" and r["last_round"] == 2 for r in data["runs"])
    est = data["estimate"]
    assert set(est["arms"]) == {"broadcast", "gossip", "llm"} and est["runs"] == 6
    assert est["arms"]["broadcast"]["usd"] == 0 and est["arms"]["llm"]["usd"] > 0
    assert est["total_usd"] == pytest.approx(2 * est["arms"]["llm"]["usd"])
    assert "estimate total:" in res.stderr                      # stdout is one JSON object
    for rid in ALL:
        assert (out / rid / "run.json").exists()

    # the same command again skips every run, with a table
    res, _ = invoke("run", EXAMPLE, "--max-rounds", 2, "--out", out, "--yes")
    assert res.exit_code == 0, res.output
    assert res.stdout.count("exists, skipping") == 6
    table = res.stdout.split("exists, skipping")[-1]
    assert "run" in table and "outcome" in table and "spend" in table
    assert all(rid in table for rid in ALL)

    # --arm/--seed narrow; --rerun writes a fresh dir
    res, data = invoke("run", EXAMPLE, "--max-rounds", 2, "--out", out, "--arm", "gossip", "--rerun",
                       "--json")
    assert res.exit_code == 0, res.output
    assert [Path(r["run_dir"]).name for r in data["runs"]] == [
        "flaggame-m1a__gossip__s1__r2", "flaggame-m1a__gossip__s2__r2"]
    assert all(r["outcome"] == "ran" for r in data["runs"])

    # a different configuration in an existing dir fails that run only
    res, data = invoke("run", EXAMPLE, "--max-rounds", 3, "--out", out, "--arm", "broadcast", "--json")
    assert res.exit_code == 1
    assert [r["outcome"] for r in data["runs"]] == ["failed", "failed"]
    assert "different configuration" in data["runs"][0]["error"]


def test_run_all_failures_continue_and_spec_errors_exit_2(tmp_path):
    res, _ = invoke("run", EXAMPLE, "--arm", "nope", "--out", tmp_path)
    assert res.exit_code == 2 and "unknown arm" in res.stderr


def test_python_run_all_summary_and_load(tmp_path):
    arms = Experiment.arms_from_yaml(EXAMPLE)
    assert list(arms) == ["broadcast", "gossip", "llm"]
    exp = arms["gossip"]
    runs = exp.run_all([1, 2], 2, out=tmp_path)
    assert [r.id for r in runs] == ["flaggame-m1a__gossip__s1", "flaggame-m1a__gossip__s2"]
    again = exp.run_all([1, 2], 2, out=tmp_path)                 # skipped: same spec on disk
    assert [r.summary() for r in again] == [r.summary() for r in runs]
    s = runs[0].summary()
    assert s["arm"] == "gossip" and s["seed"] == 1 and s["end_reason"] == "max_rounds"
    assert set(s["spend"]) == {"swarm", "measurement", "reserved", "calls"}
    assert Run.load(tmp_path, "flaggame-m1a__gossip__s1").score == runs[0].score
    assert Run.load(runs[0].dir).score == runs[0].score
    with pytest.raises(FileExistsError):
        exp.run_all([1], 3, out=tmp_path)                        # different spec, same dir
    (rerun,) = exp.run_all([1], 3, out=tmp_path, rerun=True)
    assert rerun.dir.name == "flaggame-m1a__gossip__s1__r2" and rerun.id == "flaggame-m1a__gossip__s1"

    llm = arms["llm"]
    est = llm.estimate(seed=1, max_rounds=2, seeds=[1, 2, 3])
    assert est["runs"] == 3 and est["total_usd"] == pytest.approx(3 * est["usd"])


def test_fake_spend_is_labelled_simulated(tmp_path):
    out = tmp_path / "runs"
    res, _ = invoke("run", EXAMPLE, "--max-rounds", 2, "--out", out, "--arm", "llm", "--yes")
    assert res.exit_code == 0, res.output
    row = next(line for line in res.stdout.splitlines()
               if line.startswith("flaggame-m1a__llm__s") and " ran " in line)
    assert row.rstrip().endswith("(simulated)") and "$" in row
    res, data = invoke("replay", next(out.iterdir()), "--json")
    assert data["simulated"] is True
    res, _ = invoke("replay", next(out.iterdir()))
    assert ") (simulated)" in res.stdout.splitlines()[0]


def test_getting_started_sample_labels_simulated_spend():
    from pathlib import Path

    readme = (Path(__file__).resolve().parent.parent / "docs" / "guide" / "getting-started.md").read_text()
    sample = readme.split("and ends with a table:")[1].split("```")[1]
    rows = [line for line in sample.splitlines() if line.startswith("demo__")]
    assert rows and all(line.endswith("(simulated)") for line in rows)
