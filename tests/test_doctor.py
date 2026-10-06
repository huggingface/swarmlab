"""`swarmlab doctor` in offline mode (no network)."""
import json
import sys

from typer.testing import CliRunner

from swarmlab import doctor
from swarmlab.cli import app

HF_SPEC = """\
name: d
options: {max_rounds: 1}
providers:
  hf: {type: openai_compat, params: {name: hf, pricing: {"Org/M": [0.1, 0.2, 0.1]}}}
arms:
  A:
    world: flaggame
    participants: [{type: llm, count: 2, params: {model: "hf:Org/M"}}]
"""


def by_name(results):
    return {c.name: c for c in results}


def test_offline_checks_without_specs():
    res = by_name(doctor.checks(offline=True, env={"HF_TOKEN": "x"}))
    assert res["python"].status == ("ok" if sys.version_info >= (3, 12) else "fail")
    assert res["key:hf"].status == "ok" and "HF_TOKEN is set" in res["key:hf"].detail
    assert "x" not in res["key:hf"].detail.replace("is set", "")      # values are never printed
    assert res["key:anthropic"].status == "info"                      # not needed without a spec
    assert res["reach:hf"].status == "skip" and res["reach:anthropic"].status == "skip"
    assert res["git"].status in ("ok", "warn")
    assert doctor.ok(res.values())


def test_anthropic_alias_key():
    res = by_name(doctor.checks(offline=True, env={"ANTHROPIC_KEY": "k"}))
    assert res["key:anthropic"].status == "ok" and "ANTHROPIC_KEY" in res["key:anthropic"].detail


def test_spec_needs_its_provider_key(tmp_path):
    spec = tmp_path / "d.yaml"
    spec.write_text(HF_SPEC)
    res = by_name(doctor.checks([spec], offline=True, env={}))
    assert res[f"spec:{spec}"].status == "ok" and "hf" in res[f"spec:{spec}"].detail
    assert res["key:hf"].status == "fail"
    assert not doctor.ok(res.values())
    assert by_name(doctor.checks([spec], offline=True, env={"HF_TOKEN": "t"}))["key:hf"].status == "ok"


def test_broken_spec_fails(tmp_path):
    spec = tmp_path / "bad.yaml"
    spec.write_text("name: x\narms: {A: {world: nosuchworld, participants: [scripted]}}\n")
    res = by_name(doctor.checks([spec], offline=True, env={}))
    assert res[f"spec:{spec}"].status == "fail"


def test_cli_doctor_offline(tmp_path, monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HF_TOKEN", "t")
    r = CliRunner().invoke(app, ["doctor", "--offline"])
    assert r.exit_code == 0, r.output
    assert "python" in r.stdout and "SKIP" in r.stdout and r.stdout.strip().endswith("ready")
    spec = tmp_path / "d.yaml"
    spec.write_text(HF_SPEC)
    monkeypatch.delenv("HF_TOKEN")
    r = CliRunner().invoke(app, ["doctor", str(spec), "--offline", "--json"])
    assert r.exit_code == 1
    data = json.loads(r.stdout)
    assert data["ok"] is False
    assert {c["name"]: c["status"] for c in data["checks"]}["key:hf"] == "fail"
