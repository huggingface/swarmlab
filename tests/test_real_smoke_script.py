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
    assert list(smoke.ARMS) == ["haiku", "qwen", "haiku-json"]
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
    for arm in smoke.ARMS:
        usd = float(re.search(rf"estimate {re.escape(arm)}: worst-case \$([0-9.]+)", out).group(1))
        assert usd <= smoke.ARMS[arm].get("budget", smoke.BUDGET).hard_usd
    assert "not calling any provider" in out
