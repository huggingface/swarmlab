"""`LLMAgent(system_prompt_append=...)`: one sentence added to the default prompt, in the spec only
when set; the M3 scale spec's one-sentence arm."""
from pathlib import Path

from swarmlab.participants import LLMAgent
from swarmlab.participants.llm import render_system_prompt
from swarmlab.prompts_cmd import render_prompts
from swarmlab.tools import ToolSchema

from .helpers import llm_agent_experiment
from .test_cli import invoke

REPO = Path(__file__).resolve().parents[1]
SENTENCE = "A shared message board exists."
TOOLS = [ToolSchema(name="post", description="Post a message", parameters={})]


def render(template=None, append=None):
    return render_system_prompt(template, agent="a000", role="worker", description="Find it.",
                                tools=TOOLS, append=append)


def test_append_goes_after_the_task_section_before_the_tools():
    plain, added = render(), render(append=SENTENCE + "\n")
    assert added == plain.replace("\n\nTools:\n", f"\n\n{SENTENCE}\n\nTools:\n")
    assert added.index("ends your turn.") < added.index(SENTENCE) < added.index("Tools:")


def test_custom_template_gets_the_text_at_its_end_unless_it_places_it():
    assert render("Agent {{ agent }}.", SENTENCE) == f"Agent a000.\n\n{SENTENCE}"
    placed = "{{ system_prompt_append }}\nAgent {{ agent }}."
    assert render(placed, SENTENCE) == f"{SENTENCE}\nAgent a000."
    assert render("Agent {{ agent }}.") == "Agent a000."


def test_param_in_spec_only_when_set():
    assert "system_prompt_append" not in LLMAgent(model="fake:reader").spec()["params"]
    p = LLMAgent(model="fake:reader", system_prompt_append=SENTENCE).spec()["params"]
    assert p["system_prompt_append"] == SENTENCE
    a = llm_agent_experiment(2)
    b = llm_agent_experiment(2, agent_kw={"system_prompt_append": SENTENCE})
    assert a.spec_hash(1, 2) != b.spec_hash(1, 2)
    sa, sb = render_prompts(a, 1)[0]["system"], render_prompts(b, 1)[0]["system"]
    assert sb == sa.replace("\n\nTools:\n", f"\n\n{SENTENCE}\n\nTools:\n")


def test_m3_scale_spec_has_a_one_sentence_arm(tmp_path):
    text = (REPO / "experiments" / "m3_scale_qwen9b.yaml").read_text()
    assert "AND window memory" in text  # the header names both changes of the board arm
    spec = tmp_path / "m3.yaml"  # vllm needs a server; render with the fake model instead
    spec.write_text(text.replace("vllm:Qwen/Qwen3.5-9B", "fake:reader"))
    out = {}
    for arm in ("bc-9b-256", "bc-9b-256-board", "bc-9b-256-board-full"):
        res, data = invoke("prompts", spec, "--arm", arm, "--json")
        assert res.exit_code == 0, res.output
        out[arm] = data["groups"][0]["system"]
    assert out["bc-9b-256-board-full"] == out["bc-9b-256-board"]  # same sentence, same place
    base, full = out["bc-9b-256"], out["bc-9b-256-board-full"]
    extra = full.replace(base.split("\n\nTools:")[0], "", 1).split("\n\nTools:")[0].strip()
    assert extra.startswith("A shared message board exists") and len(extra.splitlines()) == 1
    res, data = invoke("validate", spec, "--json")
    assert res.exit_code == 0, res.output
