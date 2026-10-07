"""What `run` does when a run dir holds a different spec hash (README "Existing run dirs")."""
import copy

import yaml

from swarmlab import Run

from .test_cli_run_all import invoke

SPEC = {"name": "mm", "options": {"max_rounds": 1}, "arms": {"A": {
    "world": "flaggame", "participants": [{"type": "evidence_aggregator", "count": 2}]}}}


def test_mismatch_refuses_with_both_hashes_rerun_writes_r2(tmp_path):
    doc = copy.deepcopy(SPEC)
    spec, out = tmp_path / "mm.yaml", tmp_path / "runs"
    spec.write_text(yaml.safe_dump(doc))
    assert invoke("run", spec, "--seed", 0, "--out", out)[0].exit_code == 0
    old = Run(out / "mm__A__s0").meta["spec_hash"]
    doc["arms"]["A"]["options"] = {"max_rounds": 2}
    spec.write_text(yaml.safe_dump(doc))
    res, data = invoke("run", spec, "--seed", 0, "--out", out, "--json")
    assert res.exit_code == 1 and "FileExistsError" in data["error"]
    assert old in data["error"] and "--rerun" in data["error"]
    new = data["error"].split("this spec's ")[1].split(")")[0]
    assert len(new) == 64 and new != old
    res, data = invoke("run", spec, "--seed", 0, "--out", out, "--rerun", "--json")
    assert res.exit_code == 0 and data["run_dir"].endswith("mm__A__s0__r2")
    assert data["spec_hash"] == new and Run(out / "mm__A__s0").meta["spec_hash"] == old
