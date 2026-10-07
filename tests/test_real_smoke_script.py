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
    assert list(smoke.ARMS) == ["haiku", "qwen", "haiku-json", "haiku-image"]
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
