import asyncio

import pytest

from swarmlab.blobs import BlobStore
from swarmlab.executor import RoundExecutor
from swarmlab.ids import agent_id
from swarmlab.medium.board import Board
from swarmlab.rng import derive
from swarmlab.tools import TurnCapReached
from swarmlab.world.flaggame import FlagGame

AGENTS = [agent_id(i) for i in range(4)]


def setup(tmp_path, commit="round_end", round=1, max_calls=20, allowlist=None, board=None):
    world = FlagGame()
    world.reset(derive(1, "world"), AGENTS)
    board = board or Board()
    board.commit_mode = commit
    blobs = BlobStore(tmp_path / "blobs")
    ex = RoundExecutor(run="r", round=round, world=world, board=board, blobs=blobs, agents=AGENTS,
                       commit=commit, max_calls_per_turn=max_calls,
                       topology_rng=lambda: derive(1, "topology", round), allowlist=allowlist)
    return world, board, blobs, ex


def types(ex, agent):
    return [e.type for e in ex.events(agent)]


def test_schemas_and_allowlist(tmp_path):
    _, _, _, ex = setup(tmp_path, allowlist={AGENTS[1]: {"guess"}})
    names = [s.name for s in ex.schemas(AGENTS[0])]
    assert names == ["guess", "my_status", "collective_status", "read_board", "post", "end_turn"]
    assert [s.name for s in ex.schemas(AGENTS[1])] == ["guess", "end_turn"]


async def test_round_end_post_is_pending_and_not_on_board(tmp_path):
    _, board, _, ex = setup(tmp_path)
    a = AGENTS[0]
    r1 = await ex.call(a, "post", {"text": "hi"})
    r2 = await ex.call(a, "post", {"channel": "main", "text": "again", "fields": {"k": 1}})
    assert r1.ok and r1.pending and r1.result == {"id": f"tmp-{a}-0"}
    assert r2.result == {"id": f"tmp-{a}-1"}
    assert board.buffered == []  # board untouched until commit
    assert ex.buffered_posts(a) == [("main", "hi", {}), ("main", "again", {"k": 1})]
    assert types(ex, a) == ["tool_called", "tool_returned"] * 2
    ret = ex.events(a)[1]
    assert ret.pending is True and ret.result == {"ok": True, "result": {"id": f"tmp-{a}-0"}, "error": None}


async def test_round_end_action_pending_world_unchanged(tmp_path):
    world, _, _, ex = setup(tmp_path)
    a = AGENTS[0]
    res = await ex.call(a, "guess", {"candidate": "A"})
    assert res.ok and res.pending and res.result == {"id": f"x0001-{a}-00"}
    assert world.my_status(a) == {"current_guess": None, "guesses_made": 0}
    st = await ex.call(a, "my_status", {})
    assert st.result == {"current_guess": None, "guesses_made": 0} and not st.pending
    assert [(ag, aid, act.name) for ag, aid, act in ex.buffered_actions(a)] == [(a, f"x0001-{a}-00", "guess")]


async def test_invalid_action_and_unknown_tool(tmp_path):
    _, _, _, ex = setup(tmp_path, allowlist={AGENTS[1]: {"post"}})
    bad = await ex.call(AGENTS[0], "guess", {"candidate": "ZZ"})
    assert not bad.ok and "unknown candidate" in bad.error
    assert ex.buffered_actions(AGENTS[0]) == []
    unk = await ex.call(AGENTS[0], "fly", {})
    assert not unk.ok and unk.error == "not_allowed"
    blocked = await ex.call(AGENTS[1], "guess", {"candidate": "A"})
    assert not blocked.ok and blocked.error == "not_allowed"
    bad_post = await ex.call(AGENTS[0], "post", {"text": "x", "channel": "nope"})
    assert not bad_post.ok
    assert types(ex, AGENTS[1]) == ["tool_called", "tool_returned"]


async def test_end_turn_returns_and_rejects_later_calls(tmp_path):
    _, _, _, ex = setup(tmp_path)
    a = AGENTS[2]
    assert not ex.turn_ended(a)
    done = await ex.call(a, "end_turn", {})
    assert done.ok and done.error is None and ex.turn_ended(a)
    late = await ex.call(a, "post", {"text": "late"})
    assert not late.ok and late.error == "turn_ended"
    again = await ex.call(a, "end_turn", {})
    assert not again.ok and again.error == "turn_ended"
    assert ex.buffered_posts(a) == []
    assert types(ex, a) == ["tool_called", "tool_returned"] * 3  # rejected calls are still logged
    assert ex.events(a)[3].result["error"] == "turn_ended"
    ex.end_turn_event(a, "end_turn")
    assert types(ex, a)[-1] == "turn_ended" and ex.events(a)[-1].calls == 3


async def test_cap_applies_after_end_turn(tmp_path):
    _, _, _, ex = setup(tmp_path, max_calls=2)
    a = AGENTS[0]
    await ex.call(a, "end_turn", {})
    await ex.call(a, "my_status", {})
    with pytest.raises(TurnCapReached):
        await ex.call(a, "my_status", {})


async def test_cap(tmp_path):
    _, _, _, ex = setup(tmp_path, max_calls=2)
    a = AGENTS[0]
    await ex.call(a, "my_status", {})
    await ex.call(a, "my_status", {})
    with pytest.raises(TurnCapReached):
        await ex.call(a, "my_status", {})
    last = ex.events(a)[-1]
    assert last.type == "tool_returned" and last.result["error"] == "cap"
    assert ex.calls(a) == 3


async def test_read_board_returns_content_and_logs_read(tmp_path):
    _, board, blobs, _ = setup(tmp_path)
    board.buffer_post(AGENTS[1], 1, "main", "crop:\nrg", {})
    board.commit(1, AGENTS, derive(1, "topology", 1), blobs)
    _, _, _, ex = setup(tmp_path, round=2, board=board)
    a = AGENTS[0]
    res = await ex.call(a, "read_board", {"limit": 10})
    assert res.ok and not res.pending
    (item,) = res.result["items"]
    assert item["content"] == "crop:\nrg" and item["eligible_round"] == 2
    assert types(ex, a) == ["tool_called", "read", "tool_returned"]
    assert ex.events(a)[1].delivery_ids == [item["delivery_id"]]
    again = await ex.call(a, "read_board", {})
    assert again.result == {"items": []}


async def test_immediate_applies_at_once(tmp_path):
    world, _, _, ex = setup(tmp_path, commit="immediate")
    a, b = AGENTS[0], AGENTS[1]
    res = await ex.call(a, "guess", {"candidate": "B"})
    assert res.ok and not res.pending
    assert res.result == {"id": f"x0001-{a}-00", "accepted": True, "feedback": {"recorded": True}}
    assert world.my_status(a)["current_guess"] == "B"
    post = await ex.call(a, "post", {"text": "now"})
    assert not post.pending and post.result["id"] == "p0001-0000"
    assert len(ex.committed.posts) == 1 and len(ex.committed.deliveries) == 3
    seen = await ex.call(b, "read_board", {})
    assert [i["content"] for i in seen.result["items"]] == ["now"]


async def test_concurrent_calls_keep_per_agent_state(tmp_path):
    _, _, _, ex = setup(tmp_path)

    async def agent_turn(a):
        for i in range(5):
            await ex.call(a, "post", {"text": f"{a}-{i}"})
            await asyncio.sleep(0)

    await asyncio.gather(*(agent_turn(a) for a in AGENTS))
    for a in AGENTS:
        assert [t for _, t, _ in ex.buffered_posts(a)] == [f"{a}-{i}" for i in range(5)]
        assert all(e.agent == a for e in ex.events(a))
        calls = [e.call_id for e in ex.events(a) if e.type == "tool_called"]
        assert calls == [f"c0001-{a}-{n:03d}" for n in range(1, 6)]


async def test_round_end_guess_limit_counts_buffered_guesses(tmp_path):
    from swarmlab.world.base import Action

    world, _, _, ex = setup(tmp_path)
    world.guess_limit = 2
    a = AGENTS[0]
    assert (await ex.call(a, "guess", {"candidate": "A"})).ok
    assert (await ex.call(a, "guess", {"candidate": "B"})).ok
    third = await ex.call(a, "guess", {"candidate": "C"})
    assert not third.ok and third.error == ("guess limit reached: at most 2 guesses; this one would be "
                                            "rejected at commit")
    assert [act.args["candidate"] for _, _, act in ex.buffered_actions(a)] == ["A", "B"]
    assert (await ex.call(AGENTS[1], "guess", {"candidate": "C"})).ok  # per agent
    # one guess already recorded plus one buffered also reaches the limit
    world.guesses_made[AGENTS[2]] = 1
    pend = [Action(name="guess", args={"candidate": "A"})]
    assert world.validate(AGENTS[2], pend[0]).ok
    assert not world.validate(AGENTS[2], pend[0], pending=pend).ok


async def test_immediate_guess_limit_unchanged(tmp_path):
    world, _, _, ex = setup(tmp_path, commit="immediate")
    world.guess_limit = 1
    a = AGENTS[0]
    assert (await ex.call(a, "guess", {"candidate": "A"})).result["accepted"]
    second = await ex.call(a, "guess", {"candidate": "B"})
    assert not second.ok and second.error == "guess limit reached"  # validate, as before
