"""Every tool schema the harness emits is a complete, Anthropic-strict-compatible object schema.

Regression for the 2026-10-06 smoke: Haiku rejected every request with `tools.4.custom: For
'object' type, 'additionalProperties' must be explicitly set to false` (tool 4 was `post`, sent
strict, whose nested `fields: {"type": "object"}` was open).
"""
import random

import pytest

from swarmlab import Board
from swarmlab.executor import END_TURN_SCHEMA, RoundExecutor, board_schemas
from swarmlab.providers.anthropic import build_kwargs, tool_definition
from swarmlab.providers.base import ChatMessage, ChatRequest
from swarmlab.providers.openai_compat import build_body
from swarmlab.tools import ToolSchema, strict_violations
from swarmlab.world.base import NO_ARGS, World, tool
from swarmlab.world.flaggame import FlagGame

AGENTS = ["a000", "a001"]


def anthropic_strict_check(params: dict) -> None:
    """What the Anthropic API demands of a strict tool's input_schema (raises AssertionError)."""
    assert params.get("type") == "object"
    assert strict_violations(params) == [], strict_violations(params)


def flaggame_schemas() -> list[ToolSchema]:
    world = FlagGame()
    world.reset(random.Random(1), AGENTS)
    ex = RoundExecutor(run="r", round=1, world=world, board=Board(), blobs=None, agents=AGENTS)
    return ex.schemas("a000")


def test_every_flaggame_schema_is_strict_compatible():
    schemas = flaggame_schemas()
    names = [s.name for s in schemas]
    assert names == ["guess", "my_status", "collective_status", "read_board", "post", "end_turn"]
    for s in schemas:
        p = s.parameters
        assert isinstance(p["properties"], dict) and isinstance(p["required"], list), s.name
        assert p["additionalProperties"] is False, s.name
        anthropic_strict_check(p)
        assert tool_definition(s)["strict"] is True


def test_post_no_longer_advertises_freeform_fields():
    post = next(s for s in board_schemas(Board()) if s.name == "post")
    assert set(post.parameters["properties"]) == {"channel", "text"}
    assert post.parameters["required"] == ["text"]


def test_raw_board_and_end_turn_schemas_are_already_complete():
    for s in [*board_schemas(Board()), END_TURN_SCHEMA, ToolSchema(name="x", description="",
                                                                   parameters=NO_ARGS)]:
        assert s.normalized() == s
        assert s.strict_violations() == []


def test_world_status_tools_and_tool_decorator_are_normalised():
    class W(World):
        @tool("act", "Act.", {"n": "integer"})
        def act(self, agent, n):  # pragma: no cover - never committed here
            raise NotImplementedError

        def my_status(self, agent):
            return {}

    for s in W().tool_schemas():
        assert s.strict_violations() == [], s.name


@pytest.mark.parametrize("params, expected", [
    ({}, {"type": "object", "properties": {}, "required": [], "additionalProperties": False}),
    ({"type": "object", "properties": {}},
     {"type": "object", "properties": {}, "required": [], "additionalProperties": False}),
    ({"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a", "zz"]},
     {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"],
      "additionalProperties": False}),
])
def test_normalized_fills_in_the_object_schema(params, expected):
    s = ToolSchema(name="t", description="d", parameters=params).normalized()
    assert s.parameters == expected
    assert s.normalized() == s  # idempotent


def test_normalized_recurses_into_declared_nested_objects_and_array_items():
    params = {"type": "object", "properties": {
        "pt": {"type": "object", "properties": {"x": {"type": "integer"}}},
        "pts": {"type": "array", "items": {"type": "object", "properties": {"y": {"type": "integer"}}}},
    }}
    s = ToolSchema(name="t", description="d", parameters=params).normalized()
    assert s.parameters["properties"]["pt"] == {"type": "object", "properties": {"x": {"type": "integer"}},
                                                "required": [], "additionalProperties": False}
    assert s.parameters["properties"]["pts"]["items"]["additionalProperties"] is False
    assert s.strict_violations() == []


def test_freeform_nested_object_is_left_open_and_sent_non_strict():
    """The smoke failure: a strict tool with an open nested object. Now sent without strict."""
    post_as_it_was = ToolSchema(name="post", description="Post.", parameters={
        "type": "object", "additionalProperties": False, "required": ["text"],
        "properties": {"text": {"type": "string"}, "fields": {"type": "object"}}})
    s = post_as_it_was.normalized()
    assert s.parameters["properties"]["fields"] == {"type": "object"}
    assert s.strict_violations() == ["$.fields: object without properties",
                                     "$.fields: object without a required list",
                                     "$.fields: additionalProperties is not false"]
    d = tool_definition(post_as_it_was)
    assert "strict" not in d and d["input_schema"]["additionalProperties"] is False


def test_adapters_normalise_unnormalised_schemas_before_sending():
    raw = ToolSchema(name="my_status", description="s", parameters={"type": "object", "properties": {}})
    req = ChatRequest(model="anthropic:claude-haiku-4-5", tools=[raw],
                      messages=[ChatMessage(role="user", content="hi")])
    sent = build_kwargs(req)["tools"][0]
    assert sent["strict"] is True and sent["input_schema"]["additionalProperties"] is False
    body = build_body(req.model_copy(update={"model": "hf:Qwen/Qwen3.5-9B"}))
    assert body["tools"][0]["function"]["parameters"]["required"] == []


def test_smoke_request_tools_index_4_was_post():
    """The stored smoke request (blob 976a2ab9...) listed these tools; index 4 was `post`."""
    sent = [t.name for t in flaggame_schemas()]
    assert sent.index("post") == 4
