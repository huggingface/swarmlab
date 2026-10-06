"""Identifier types. All are plain strings with a NewType for readability."""
from typing import NewType

AgentId = NewType("AgentId", str)
RunId = NewType("RunId", str)
PostId = NewType("PostId", str)
ActionId = NewType("ActionId", str)
CallId = NewType("CallId", str)
DeliveryId = NewType("DeliveryId", str)


def agent_id(index: int) -> AgentId:
    """`a` + zero-padded index: a000, a001, ..."""
    return AgentId(f"a{index:03d}")


def run_id(experiment: str, arm: str, seed: int) -> RunId:
    return RunId(f"{experiment}__{arm}__s{seed}")


def fork_run_id(parent: RunId, at_round: int, n: int) -> RunId:
    return RunId(f"{parent}__f{at_round}_{n}")
