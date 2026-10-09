"""`LLMAgent(system_prompt_append=...)`: one sentence added to the default prompt, in the spec only
when set."""
from swarmlab.participants import LLMAgent
from swarmlab.participants.llm import render_system_prompt
from swarmlab.prompts_cmd import render_prompts
from swarmlab.tools import ToolSchema

from .helpers import llm_agent_experiment

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
