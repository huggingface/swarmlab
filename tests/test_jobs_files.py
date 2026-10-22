"""`swarmlab job run` stages the files a spec references (`file:<path>` participant params) and
rewrites the job spec to the copies; `spec_hash` hashes such params by content. Offline."""
import json
from pathlib import Path

import pytest
import yaml

from swarmlab import Experiment
from swarmlab.jobs import launch as jl
from swarmlab.spec import SpecError, file_refs_by_content, load_experiment_yaml, spec_hash

from .test_jobs import COMMIT, NOW, vllm_spec
from .test_jobs_api import KEYS, solo_doc, write

TEMPLATES = {"system_template": "You are {{ agent }}.", "instance_template": "Go. {{ guidance }}"}


def plan(spec, **kw):
    return jl.plan_job(spec, now=NOW, git=lambda: (COMMIT, False), environ=KEYS, **kw)


def msa_doc(tmp_path, guidance, templates):
    doc = solo_doc()
    doc["roles"]["solo"]["tools"] = ["bash"]
    doc["arms"]["solo"]["participants"] = [{"type": "mini_swe_agent", "count": 1, "role": "solo",
        "params": {"model": "anthropic:claude-haiku-5-5", "templates": templates,
                   "guidance": guidance, "step_limit": 1000}}]
    return doc


def files(tmp_path):
    (tmp_path / "a").mkdir()
    lead = tmp_path / "a" / "piv-lead.txt"
    lead.write_text("Delegate early.\n")
    tpl = tmp_path / "a" / "tpl.yaml"
    tpl.write_text(yaml.safe_dump(TEMPLATES))
    return lead, tpl


def test_two_references_are_staged_and_rewritten(tmp_path):
    lead, tpl = files(tmp_path)
    p = plan(write(tmp_path, msa_doc(tmp_path, f"file:{lead}", f"file:{tpl}")))
    assert sorted((f["name"], f["size"], f["ref"]) for f in p.staged_files) == [
        ("piv-lead.txt", 16, f"file:{lead}"), ("tpl.yaml", tpl.stat().st_size, f"file:{tpl}")]
    job = yaml.safe_load(p.job_spec_yaml)
    params = job["arms"]["solo"]["participants"][0]["params"]
    assert params["guidance"] == "file:/opt/swarmlab/stage/files/piv-lead.txt"
    assert params["templates"] == "file:/opt/swarmlab/stage/files/tpl.yaml"
    assert str(lead) not in p.job_spec_yaml and str(lead) in p.orig_spec_yaml
    d = jl.stage(p, tmp_path / "st", build=None)
    assert (d / "files" / "piv-lead.txt").read_text() == "Delegate early.\n"
    assert (d / "files" / "tpl.yaml").read_text() == tpl.read_text()
    recorded = json.loads((d / "plan.json").read_text())["staged_files"]
    assert {f["name"]: f["sha256"] for f in recorded} == {f["name"]: f["sha256"] for f in p.staged_files}
    assert "staged file: file:" in jl.describe(p)
    lead.write_text("changed")
    with pytest.raises(SpecError, match="changed since the plan"):
        jl.stage(p, tmp_path / "st2", build=None)


def test_duplicate_basenames_with_different_contents_are_refused(tmp_path):
    lead, tpl = files(tmp_path)
    (tmp_path / "b").mkdir()
    other = tmp_path / "b" / "piv-lead.txt"
    other.write_text("Something else.\n")
    doc = msa_doc(tmp_path, f"file:{lead}", f"file:{tpl}")
    doc["arms"]["solo"]["participants"][0]["params"]["extra"] = {"note": f"file:{other}"}
    with pytest.raises(SpecError, match="share the name 'piv-lead.txt'"):
        plan(write(tmp_path, doc))
    other.write_text("Delegate early.\n")   # same content: staged once
    p = plan(write(tmp_path, doc))
    assert sorted(f["name"] for f in p.staged_files) == ["piv-lead.txt", "tpl.yaml"]
    with pytest.raises(SpecError, match="cannot read"):
        plan(write(tmp_path, msa_doc(tmp_path, f"file:{tmp_path}/missing.txt", f"file:{tpl}")))


def test_staged_paths_hash_like_the_local_run(tmp_path):
    lead, tpl = files(tmp_path)
    spec = write(tmp_path, msa_doc(tmp_path, f"file:{lead}", f"file:{tpl}"))
    p = plan(spec)
    st = jl.stage(p, tmp_path / "st", build=None)
    # the job's spec, with its stage dir where the job has it
    job_yaml = p.job_spec_yaml.replace(jl.STAGE_LOCAL, str(st))
    (tmp_path / "job.yaml").write_text(job_yaml)
    local = Experiment.from_yaml(spec, "solo")
    job = Experiment.from_yaml(tmp_path / "job.yaml", "solo")
    def run_spec(e):
        return e.to_spec(**e._options(0, 1).model_dump())

    assert run_spec(local).participants != run_spec(job).participants   # the paths differ ...
    assert local.spec_hash(0, 1) == job.spec_hash(0, 1)                  # ... the hash does not
    lead.write_text("Edited.\n")
    assert Experiment.from_yaml(spec, "solo").spec_hash(0, 1) != job.spec_hash(0, 1)
    assert file_refs_by_content({"x": [f"file:{tmp_path}/nope"]}) == {"x": [f"file:{tmp_path}/nope"]}
    assert spec_hash(run_spec(local)) == local.spec_hash(0, 1)


def test_guidance_file_is_read_at_render_time(tmp_path):
    from swarmlab.participants.minisweagent import MiniSweAgent

    lead, _ = files(tmp_path)
    agent = MiniSweAgent(model="anthropic:claude-haiku-5-5", templates="worker",
                         guidance=f"file:{lead}")
    view = type("V", (), {"description": "", "turn": None})()
    assert agent.template_vars(view, [])["guidance"] == "Delegate early."  # resolve_guidance (#33) strips the text


def test_a_spec_without_references_is_unchanged(tmp_path):
    p = plan(vllm_spec(tmp_path), model="Qwen/Qwen3.5-9B")
    assert p.staged_files == []
    doc = load_experiment_yaml(tmp_path / "spec.yaml")
    assert p.job_spec_yaml == jl.dump_yaml(jl.job_spec_doc(doc, "Qwen/Qwen3.5-9B", arms=["bc"]))
    api = plan(write(tmp_path, solo_doc(), "api.yaml"))
    assert api.staged_files == [] and jl.dump_yaml(jl.job_spec_doc(
        load_experiment_yaml(tmp_path / "api.yaml"), None, arms=["solo"])) == api.job_spec_yaml
    assert not (jl.stage(api, tmp_path / "st", build=None) / "files").exists()


# ---- a participant that opens its file: param at construction (as mini_swe_agent's guidance on
# the piv-prompts branch does): planning must not build it from the in-job paths ----------------
from swarmlab.participants.scripted import EvidenceAggregator

OPENED: list[str] = []


class OpensFile(EvidenceAggregator):
    """Test participant: reads its `notes: file:<path>` param in __init__."""

    entry_point = None

    def __init__(self, notes: str) -> None:
        OPENED.append(notes)
        self.notes_text = Path(notes[len("file:"):]).read_text()


def opens_doc(notes):
    return {"name": "opens", "seeds": [0], "options": {"max_rounds": 1},
            "arms": {"a": {"world": "flaggame", "participants": [
                {"type": "tests.test_jobs_files:OpensFile", "count": 3,
                 "params": {"notes": f"file:{notes}"}}]}}}


@pytest.mark.parametrize("mode", ["api", "vllm"])
def test_constructor_reads_local_paths_and_the_stage_carries_the_rewrite(tmp_path, mode,
                                                                         monkeypatch):
    from typer.testing import CliRunner

    from swarmlab.cli import app

    notes = tmp_path / "piv-solo.txt"
    notes.write_text("Work alone.\n")
    doc = opens_doc(notes)
    if mode == "vllm":   # one llm agent so the job has a model to serve
        doc["arms"]["a"]["participants"].append(
            {"type": "llm", "count": 1, "params": {"model": "vllm:Qwen/Qwen3.5-9B"}})
    OPENED.clear()
    model = "Qwen/Qwen3.5-9B" if mode == "vllm" else None
    spec = write(tmp_path, doc)
    p = plan(spec, **({"model": model} if model else {}))
    assert OPENED and all(o == f"file:{notes}" for o in OPENED)   # never the in-job path
    st = jl.stage(p, tmp_path / "st", build=None)
    staged = yaml.safe_load((st / "spec.yaml").read_text())
    assert staged["arms"]["a"]["participants"][0]["params"]["notes"] == \
        "file:/opt/swarmlab/stage/files/piv-solo.txt"
    assert (st / "files" / "piv-solo.txt").read_text() == "Work alone.\n"
    # inside the job the stage dir is /opt/swarmlab/stage: there `validate` reads the copy
    in_job = tmp_path / "in_job.yaml"
    in_job.write_text((st / "spec.yaml").read_text().replace(jl.STAGE_LOCAL, str(st)))
    OPENED.clear()
    res = CliRunner().invoke(app, ["validate", str(in_job)])
    assert res.exit_code == 0, res.output
    assert OPENED and all(o == f"file:{st}/files/piv-solo.txt" for o in OPENED)
    local_spec = tmp_path / "local.yaml"   # the job spec with the local paths
    local_spec.write_text(jl.dump_yaml(jl.job_spec_doc(load_experiment_yaml(spec), model, arms=["a"])))
    local, job = Experiment.from_yaml(local_spec, "a"), Experiment.from_yaml(in_job, "a")
    assert local.spec_hash(0, 1) == job.spec_hash(0, 1)
