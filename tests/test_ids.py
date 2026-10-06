from swarmlab.ids import AgentId, agent_id, fork_run_id, run_id


def test_agent_id_padding():
    assert agent_id(0) == "a000"
    assert agent_id(7) == "a007"
    assert agent_id(123) == "a123"
    assert agent_id(1000) == "a1000"
    assert isinstance(agent_id(1), str)
    assert sorted(agent_id(i) for i in (10, 2, 1)) == ["a001", "a002", "a010"]


def test_run_ids():
    rid = run_id("flag", "A", 3)
    assert rid == "flag__A__s3"
    assert fork_run_id(rid, 4, 1) == "flag__A__s3__f4_1"


def test_newtypes_are_str():
    assert AgentId("a001") == "a001"
