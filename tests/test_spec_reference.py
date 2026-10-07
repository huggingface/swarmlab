"""`swarmlab spec-reference`, README's generated section, and readable shape errors."""
from pathlib import Path

import pytest
from typer.testing import CliRunner

from swarmlab.cli import app
from swarmlab.spec import SpecError, validate_experiment_doc
from swarmlab.spec_reference import DESCRIPTIONS, readme_section, spec_reference

README = Path(__file__).resolve().parent.parent / "README.md"
runner = CliRunner()


def doc(**arm):
    base = {"world": "flaggame", "participants": ["silent"]}
    return {"name": "x", "arms": {"g": {**base, **arm}}}


def test_readme_section_matches_the_command():
    res = runner.invoke(app, ["spec-reference"])
    assert res.exit_code == 0
    assert readme_section(README.read_text()) == res.output.rstrip("\n") == spec_reference()


def test_every_model_field_is_described():
    for model, desc in DESCRIPTIONS.items():
        assert set(model.model_fields) == set(desc), model.__name__


def test_reference_lists_the_shapes_the_dogfood_had_to_guess():
    text = spec_reference()
    for needle in ("`arms.NAME.medium.topology`: `{type: NAME, params: {...}}`",
                   "`arms.NAME.medium.policies`: `[{type: NAME, params: {...}}, ...]`",
                   "`arms.NAME.medium.registry`", "`arms.NAME.medium.claim_policy`",
                   "`arms.NAME.participants`: `[{type, count, params, role}, ...]`",
                   "`arms.NAME.probes`", "`interventions`", "`roles`", "`arms.NAME.budget`",
                   "`budget.total_usd`", "`options.max_rounds`", "`seeds`", "`providers`"):
        assert needle in text, needle
    assert "arms.NAME.budget.total_usd" not in text


def test_readme_command_rewrites_the_section(tmp_path):
    p = tmp_path / "R.md"
    p.write_text("intro\n<!-- spec-reference:start -->\nstale\n<!-- spec-reference:end -->\nend\n")
    assert runner.invoke(app, ["spec-reference", "--readme", str(p)]).exit_code == 0
    text = p.read_text()
    assert readme_section(text) == spec_reference() and text.endswith("end\n")


@pytest.mark.parametrize("medium,expect", [
    ({"topology": "gossip", "topology_params": {"k": 1}},
     ["arms.g.medium.topology_params: unknown key 'topology_params'",
      "topology: {type: gossip, params: {k: 1}}"]),
    ({"topology": {"type": "gossip", "k": 1}},
     ["arms.g.medium.topology.k: unknown key 'k'",
      "arms.g.medium.topology expects {type: NAME, params: {...}} or a bare NAME string"]),
    ({"topology": ["gossip"]},
     ["arms.g.medium.topology:", "expects {type: NAME, params: {...}}"]),
])
def test_wrong_topology_shape_says_the_right_shape(medium, expect):
    with pytest.raises(SpecError) as e:
        validate_experiment_doc(doc(medium=medium))
    for needle in expect:
        assert needle in str(e.value)
    assert "swarmlab spec-reference" in str(e.value)


def test_wrong_participant_key_names_the_group_shape():
    with pytest.raises(SpecError, match=r"arms\.g\.participants\.0 expects a mapping with keys "
                                         r"type, count, params, role"):
        validate_experiment_doc(doc(participants=[{"type": "silent", "n": 3}]))


def test_shorthand_is_not_accepted():
    with pytest.raises(SpecError):
        validate_experiment_doc(doc(medium={"topology": "gossip", "topology_params": {"k": 1}}))
