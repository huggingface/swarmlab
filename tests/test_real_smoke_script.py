"""tools/real_smoke.py without spending money: arm configuration and the estimate guard."""
import importlib.util
import re
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "real_smoke.py"


@pytest.fixture(scope="module")
def smoke():
    spec = importlib.util.spec_from_file_location("real_smoke", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_arms(smoke):
    assert list(smoke.ARMS) == ["haiku", "qwen", "haiku-json", "haiku-image", "gemma-image",
                                "gemma-manager"]
    assert smoke.DEFAULT_ARMS == ("haiku", "qwen", "haiku-json")
    qwen = smoke.experiment("qwen")
    assert all(p.extra == {"chat_template_kwargs": {"enable_thinking": False}} for p in qwen.participants)
    hj = smoke.experiment("haiku-json")
    assert len(hj.participants) == 2 and all(p.tool_protocol == "json" for p in hj.participants)
    assert hj.budget.hard_usd <= smoke.BUDGET.hard_usd


def test_dry_run_prints_estimate_under_the_cap(smoke, monkeypatch, capsys):
    monkeypatch.delenv("SWARMLAB_REAL", raising=False)
    monkeypatch.setattr(sys, "argv", ["real_smoke.py"])
    assert smoke.main() == 0
    out = capsys.readouterr().out
    total = float(re.search(r"estimate total: \$([0-9.]+)", out).group(1))
    assert 0 < total < smoke.TOTAL_CAP_USD
    for arm in smoke.DEFAULT_ARMS:
        usd = float(re.search(rf"estimate {re.escape(arm)}: worst-case \$([0-9.]+)", out).group(1))
        assert usd <= smoke.ARMS[arm].get("budget", smoke.BUDGET).hard_usd
    assert "not calling any provider" in out


def test_haiku_image_arm(smoke):
    exp = smoke.experiment("haiku-image")
    assert exp.world.modality == "image" and not exp.world.image_text_hint
    assert len(exp.participants) == 4 and smoke.ARMS["haiku-image"]["rounds"] == 3
    assert all(p.model == smoke.HAIKU and p.max_calls == 5 and p.max_tokens == 1024
               for p in exp.participants)
    assert exp.budget.hard_usd == 0.50
    extra = smoke.image_overhead_tokens(exp)
    assert 500 < extra < 1500  # nine ~100-token images replace ~230 tokens of text grids
    est = smoke.estimate("haiku-image", exp)
    assert est["prompt_tokens"] == 3000 + extra and est["prompt_growth"] == extra
    assert est["usd"] <= 0.50
    # images cost more than the text arm's flat planning prompt
    assert est["usd"] / 5 > smoke.estimate("haiku", smoke.experiment("haiku"))["usd"] / 6


def test_dry_run_image_arm_alone(smoke, monkeypatch, capsys):
    monkeypatch.delenv("SWARMLAB_REAL", raising=False)
    monkeypatch.setattr(sys, "argv", ["real_smoke.py", "--arms", "haiku-image"])
    assert smoke.main() == 0
    out = capsys.readouterr().out
    usd = float(re.search(r"estimate haiku-image: worst-case \$([0-9.]+)", out).group(1))
    assert 0 < usd <= 0.50 and "estimate haiku:" not in out
    assert "not calling any provider" in out
    monkeypatch.setattr(sys, "argv", ["real_smoke.py", "--arms", "nope"])
    assert smoke.main() == 2


def test_visual_mentions(smoke):
    vm = smoke.visual_mentions([
        "My crop: red band on top, blue stripe below",
        "crop:\nrrgg\nrrgg",
        "I guess C",
        "The left half is Green",
    ])
    assert vm == {"posts": 4, "colour_words": 2, "layout_words": 2, "letter_rows": 1, "visual": 2}


def test_gemma_arms(smoke):
    from swarmlab.medium.topology import Broadcast, Star
    from swarmlab.roles import role_name

    for arm in ("gemma-image", "gemma-manager"):
        exp = smoke.experiment(arm)
        assert exp.world.modality == "image" and len(exp.participants) == 4
        assert smoke.ARMS[arm]["rounds"] == 3 and exp.budget.hard_usd == 0.50
        assert all(p.model == "hf:google/gemma-4-26B-A4B-it:deepinfra" and p.max_calls == 5
                   for p in exp.participants)
        assert exp.providers["hf"].pricing[smoke.GEMMA_ID] == smoke.GEMMA_PRICING
        assert smoke.estimate(arm, exp)["usd"] <= 0.50
    image, mgr = smoke.experiment("gemma-image"), smoke.experiment("gemma-manager")
    assert isinstance(image.medium.topology, Broadcast) and not image.world.blind_agents
    assert isinstance(mgr.medium.topology, Star) and mgr.medium.topology.center == "a000"
    assert mgr.world.blind_agents == 1
    assert [role_name(p) for p in mgr.participants] == ["manager", None, None, None]
    assert "gemma-image" not in smoke.DEFAULT_ARMS and "gemma-manager" not in smoke.DEFAULT_ARMS


def test_dry_run_gemma_arms(smoke, monkeypatch, capsys):
    monkeypatch.delenv("SWARMLAB_REAL", raising=False)
    monkeypatch.setattr(sys, "argv", ["real_smoke.py", "--arms", "gemma-image,gemma-manager"])
    assert smoke.main() == 0
    out = capsys.readouterr().out
    assert "estimate gemma-manager: worst-case" in out and "not calling any provider" in out


def test_manager_report_on_a_fake_run(smoke, tmp_path, capsys):
    """The gemma-manager report runs on a finished run (fake reader stands in for the model)."""
    from swarmlab.participants import LLMAgent
    from swarmlab.roles import assign
    from swarmlab.worlds import FlagGame

    exp = smoke.experiment("gemma-manager")
    exp.world = FlagGame(n_candidates=8, modality="image", image_text_hint=True, blind_agents=1)
    exp.participants = [LLMAgent(model="fake:reader", max_calls=5) for _ in range(4)]
    assign(exp.participants[0], "manager")
    exp.providers = None
    run = exp.run(seed=1, max_rounds=2, out=tmp_path)
    smoke.report("gemma-manager", run)
    out = capsys.readouterr().out
    assert "manager " in out and "member posts delivered to it" in out
