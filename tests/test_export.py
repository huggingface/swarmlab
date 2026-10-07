"""M4 acceptance 1: export of a finished fake-provider run (docs/INTERFACE-M4.md §1)."""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from swarmlab import Run, export
from swarmlab.cli import app
from swarmlab.participants import LLMAgent
from swarmlab.probes import BeliefProbe

from .helpers import Chatter, flag_experiment, llm_agent_experiment
from .pi_session import validate_file


def _events(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]


def _events_of(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _expected_rows(evs: list[dict]) -> dict[str, int]:
    c = Counter(e["type"] for e in evs)
    covered = {"turn_started", "turn_ended", "tool_called", "tool_returned", "post", "delivery",
               "read", "action_committed", "inference_attempt", "inference_response", "probe",
               "metric", "intervention", "round_started", "round_committed", "budget",
               "snapshot", "run_started", "run_ended"}
    return {
        "turns": c["turn_ended"], "tool_calls": c["tool_called"], "posts": c["post"],
        "deliveries": c["delivery"],
        "reads": sum(max(1, len(e["delivery_ids"])) for e in evs if e["type"] == "read"),
        "actions": c["action_committed"], "inference": c["inference_attempt"],
        "probes": c["probe"], "metrics": c["metric"], "interventions": c["intervention"],
        "rounds": len({e["round"] for e in evs if e["type"] in ("round_started", "round_committed")}),
        "run": 1, "other": sum(n for t, n in c.items() if t not in covered),
    }


@pytest.fixture(scope="module")
def llm_run(tmp_path_factory) -> Run:
    """3 fake LLM agents (full memory) + 1 scripted chatter, belief probe, 3 rounds."""
    exp = llm_agent_experiment(3, name="exp-llm", probes=[BeliefProbe()])
    exp.participants.append(Chatter())
    out = tmp_path_factory.mktemp("runs")
    return exp.run(seed=3, max_rounds=3, out=out)


def _check_export(run_dir: Path, out: Path) -> dict:
    evs = _events(run_dir)
    doc = json.loads((out / "run.json").read_text())
    assert doc["export_schema"] == export.EXPORT_SCHEMA
    expected = _expected_rows(evs)
    expected["discarded_inference"] = sum(
        1 for e in _events_of(run_dir / "discarded.jsonl") if e["type"] == "inference_attempt")
    for fam in export.FAMILIES:
        table = pq.read_table(out / "tables" / f"{fam}.parquet")
        assert table.column_names[:6] == list(export.KEY_COLUMNS), fam
        assert table.schema.equals(export.TABLE_SCHEMAS[fam]), fam
        assert table.num_rows == expected[fam], (fam, table.num_rows, expected[fam])
        assert doc["tables"][fam]["rows"] == expected[fam]
    assert (out / "raw" / "events.jsonl").read_bytes() == (run_dir / "events.jsonl").read_bytes()
    for f in sorted((out / "sessions").glob("*.jsonl")):
        assert validate_file(f) == [], f
    return doc


def _session(out: Path, agent: str) -> list[dict]:
    lines = (out / "sessions" / f"{agent}.jsonl").read_text().splitlines()
    return [json.loads(x)["message"] for x in lines[1:]]


def _memory(run: Run, agent: str) -> list[dict]:
    from swarmlab.snapshot import SnapshotStore

    store = SnapshotStore(run.dir)
    manifest = store.latest()
    p = LLMAgent(model="fake:reader")
    p.restore(store.load(manifest)[f"participant:{agent}"])
    return [m.model_dump(mode="json") for m in p.probe_context()]


def _flat(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text") or "" for b in content)


def test_export_tables_sessions_and_raw(llm_run, tmp_path):
    out = llm_run.export(tmp_path / "export")
    doc = _check_export(llm_run.dir, out)
    assert doc["run_id"] == llm_run.id and doc["score"] == llm_run.score
    assert doc["end_reason"] == "max_rounds"
    assert set(doc["metrics_final"]) == set(llm_run.metrics)
    assert doc["blobs"]["included"] == "all"
    assert (out / "raw" / "run.json").exists() and (out / "raw" / "snapshots").is_dir()
    turns = pq.read_table(out / "tables" / "turns.parquet").to_pylist()
    assert {t["agent"] for t in turns} == {"a000", "a001", "a002", "a003"}
    inf = pq.read_table(out / "tables" / "inference.parquet").to_pylist()
    assert all(r["request"] and r["response"] for r in inf)  # small blobs are inlined
    assert {r["category"] for r in inf} == {"swarm", "measurement"}
    assert json.loads(inf[0]["request"])["model"] == "fake:reader"
    dl = pq.read_table(out / "tables" / "deliveries.parquet").to_pylist()
    posts = {p["post_id"]: p["text"] for p in
             pq.read_table(out / "tables" / "posts.parquet").to_pylist()}
    assert dl and all(r["content"] == posts[r["post_id"]] for r in dl)
    probes = pq.read_table(out / "tables" / "probes.parquet").to_pylist()
    assert probes and all(r["question"] for r in probes if r["question_hash"])


def test_llm_session_matches_the_agent_memory(llm_run, tmp_path):
    out = llm_run.export(tmp_path / "export")
    for agent in ("a000", "a001", "a002"):
        msgs = _session(out, agent)
        mem = _memory(llm_run, agent)
        assert msgs[0]["role"] == "system" and msgs[0]["content"] == mem[0]["content"]
        assert {t["name"] for t in msgs[0]["toolsAdded"]} >= {"guess", "end_turn"}
        role = {"tool": "toolResult"}
        assert [m["role"] for m in msgs] == [role.get(m["role"], m["role"]) for m in mem]
        got = [_flat(m["content"]) for m in msgs if m["role"] in ("user", "toolResult")]
        want = [_flat(m["content"]) for m in mem if m["role"] in ("user", "tool")]
        assert got == want
        users = [m for m in msgs if m["role"] == "user"]
        assert [_flat(u["content"]).split("\n")[0] for u in users] == ["Round 1.", "Round 2.",
                                                                       "Round 3."]
        asst = [m for m in msgs if m["role"] == "assistant"]
        assert all(a["provider"] == "fake" and a["usage"]["totalTokens"] > 0 for a in asst)


def test_window_memory_session(tmp_path):
    exp = llm_agent_experiment(2, name="exp-window", agent_kw={"memory": "window",
                                                               "window_rounds": 1})
    run = exp.run(seed=1, max_rounds=4, out=tmp_path / "runs")
    out = run.export(tmp_path / "export")
    _check_export(run.dir, out)
    msgs = _session(out, "a000")
    users = [_flat(m["content"]).split("\n")[0] for m in msgs if m["role"] == "user"]
    assert users == [f"Round {r}." for r in range(1, 5)]  # nothing duplicated by the window
    calls = [b["id"] for m in msgs if m["role"] == "assistant" for b in m["content"]
             if b["type"] == "toolCall"]
    results = [m["toolCallId"] for m in msgs if m["role"] == "toolResult"]
    assert calls == results


def test_scripted_session(tmp_path):
    run = flag_experiment(4, name="exp-scripted").run(seed=2, max_rounds=3, out=tmp_path / "runs")
    out = export.export_run(run.dir)
    assert out == run.dir / "export"
    _check_export(run.dir, out)
    evs = _events(run.dir)
    for agent in ("a000", "a003"):
        msgs = _session(out, agent)
        n_calls = sum(1 for e in evs if e["type"] == "tool_called" and e["agent"] == agent)
        assert sum(1 for m in msgs if m["role"] == "toolResult") == n_calls
        assert sum(1 for m in msgs if m["role"] == "user") == 3
        assert all(m["provider"] == "scripted" for m in msgs if m["role"] == "assistant")


def test_reexport_is_deterministic(llm_run, tmp_path):
    a = llm_run.export(tmp_path / "a")
    b = llm_run.export(tmp_path / "b")
    files = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
    assert files == sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file())
    for f in files:
        assert (a / f).read_bytes() == (b / f).read_bytes(), f
    assert export.is_current(llm_run.dir, a)


def test_oversize_blobs_and_partial_raw(llm_run, tmp_path, monkeypatch):
    monkeypatch.setattr(export, "INLINE_LIMIT", 200)
    monkeypatch.setattr(export, "FULL_BLOBS_LIMIT", 1)
    out = llm_run.export(tmp_path / "export")
    doc = json.loads((out / "run.json").read_text())
    assert doc["blobs"]["included"] == "snapshots+oversize"
    inf = pq.read_table(out / "tables" / "inference.parquet").to_pylist()
    big = [r["request"] for r in inf if r["request"].startswith("sha256:")]
    assert big
    for cell in big:
        sha = cell.removeprefix("sha256:")
        assert (out / "raw" / "blobs" / sha[:2] / sha).exists()
    assert not (out / "raw" / "blobs" / "cache").exists()


def test_unknown_and_intervention_events(llm_run, tmp_path):
    d = tmp_path / "copy"
    shutil.copytree(llm_run.dir, d)
    extra = [
        {"seq": 9001, "run": llm_run.id, "round": 2, "agent": "a000", "ts": 1.0,
         "type": "intervention", "intervention": "mute1", "op": "mute", "ok": True,
         "affected": ["a000"]},
        {"seq": 9002, "run": llm_run.id, "round": 2, "agent": "a001", "ts": 1.0,
         "type": "overflow", "policy": "drop_oldest", "dropped_rounds": [1]},
    ]
    with open(d / "events.jsonl", "a") as f:
        f.writelines(json.dumps(e) + "\n" for e in extra)
    out = export.export_run(d, tmp_path / "export")
    _check_export(d, out)
    iv = pq.read_table(out / "tables" / "interventions.parquet").to_pylist()
    assert iv[0]["op"] == "mute" and json.loads(iv[0]["affected"]) == ["a000"]
    other = pq.read_table(out / "tables" / "other.parquet").to_pylist()
    assert [o["type"] for o in other] == ["overflow"]


def test_cli_export(llm_run, tmp_path):
    res = CliRunner().invoke(app, ["export", str(llm_run.dir), "--out", str(tmp_path / "e"),
                                   "--json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["export"] == str(tmp_path / "e")
    assert data["tables"]["turns"] == 12
    assert data["sessions"] == 4


def test_discarded_rounds_are_exported_with_their_spend(tmp_path):
    """A hard-ceiling abort discards the round in flight; its calls and spend stay visible."""
    from swarmlab.spec import Budget

    exp = llm_agent_experiment(4, name="exp-disc", pricing={"*": (10.0, 50.0, 1.0)},
                               agent_kw={"max_tokens": 64, "max_calls": 2},
                               budget=Budget(hard_usd=0.06))
    run = exp.run(seed=1, max_rounds=3, out=tmp_path / "runs")
    assert run.end_reason == "hard_ceiling"
    run = run.resume(budget=Budget(hard_usd=5.0))  # moves the aborted round to discarded.jsonl
    assert run.end_reason == "max_rounds"
    out = run.export()
    doc = _check_export(run.dir, out)
    disc = pq.read_table(out / "tables" / "discarded_inference.parquet").to_pylist()
    assert disc and doc["tables"]["discarded_inference"]["rows"] == len(disc)
    assert len({r["round"] for r in disc}) == 1
    assert doc["spend_discarded_usd"] == pytest.approx(sum(r["charged_usd"] for r in disc))
    assert doc["spend_discarded_usd"] > 0
    assert doc["spend_discarded_usd"] == pytest.approx(export.discarded_spend(run.dir))
    # the ledger = what the kept log charged + what the discarded round charged
    kept = sum(r["cost_usd"] or 0 for r in pq.read_table(out / "tables" / "inference.parquet")
               .to_pylist() if not r["cached"])
    ledger = doc["spend"]["swarm"] + doc["spend"]["measurement"]
    assert kept + doc["spend_discarded_usd"] == pytest.approx(ledger)


def test_report_reconciles_spend_with_the_ledger(tmp_path):
    from swarmlab.report import build_report, spend_lines
    from swarmlab.spec import Budget

    exp = llm_agent_experiment(4, name="exp-disc", pricing={"*": (10.0, 50.0, 1.0)},
                               agent_kw={"max_tokens": 64, "max_calls": 2},
                               budget=Budget(hard_usd=0.06))
    run = exp.run(seed=1, max_rounds=3, out=tmp_path / "runs").resume(budget=Budget(hard_usd=5.0))
    total = run.spend["swarm"] + run.spend["measurement"]
    line = spend_lines([run])[0]  # what the report prints for real (non-fake) runs
    assert f"${total:.3f}" in line and f"${export.discarded_spend(run.dir):.3f}" in line
    # these runs use fake: models, so the report sets them apart and counts none of their spend
    assert "over the 0 real run(s): $0.000; 1 simulated" in build_report(tmp_path / "runs")


def _users(msgs: list[dict]) -> list[str]:
    return [_flat(m["content"]).split("\n")[0] for m in msgs if m["role"] == "user"]


def test_unreadable_request_blob_does_not_duplicate_rounds(llm_run, tmp_path):
    """Field notes item 9: a request blob that cannot be read (EIO on a bucket mount, or gone)
    used to reset the overlap, so the next request was emitted in full: a second `Round 1.`
    marker with round-2 content under it. Rounds stay in order and nothing is repeated."""
    d = tmp_path / "copy"
    shutil.copytree(llm_run.dir, d)
    reqs = [e for e in _events(d) if e["type"] == "inference_attempt"
            and e["agent"] == "a000" and e["category"] == "swarm"]
    lost = next(e for e in reqs if e["round"] == 2)
    (d / "blobs" / lost["request_hash"][:2] / lost["request_hash"]).unlink()
    out = export.export_run(d, tmp_path / "export")
    for f in sorted((out / "sessions").glob("*.jsonl")):
        assert validate_file(f) == [], f  # includes: round markers strictly increasing
    msgs = _session(out, "a000")
    assert _users(msgs) == ["Round 1.", "Round 2.", "Round 3."]
    full = _session(llm_run.export(tmp_path / "full"), "a000")
    # the same conversation as with every blob present, in the same order
    assert [m["role"] for m in msgs] == [m["role"] for m in full]
    assert [_flat(m["content"]) if m["role"] != "assistant" else m["content"] for m in msgs] == \
        [_flat(m["content"]) if m["role"] != "assistant" else m["content"] for m in full]


def test_validator_rejects_a_repeated_round_marker():
    from .pi_session import validate_lines

    head = json.dumps({"type": "session", "version": 3, "id": "r/a000",
                       "timestamp": "2026-10-07T00:00:00.000Z", "cwd": "runs/r", "harness": "swarmlab"})

    def user(i, text, parent):
        return json.dumps({"type": "message", "id": f"e{i}", "parentId": parent,
                           "timestamp": "2026-10-07T00:00:00.000Z",
                           "message": {"role": "user", "timestamp": 0,
                                       "content": [{"type": "text", "text": text}]}})

    ok = [head, user(1, "Round 1.\nx", None), user(2, "Round 2.\ny", "e1")]
    assert validate_lines(ok) == []
    bad = [*ok, user(3, "Round 1.\nz", "e2")]
    assert any("round marker 'Round 1.' after 'Round 2.'" in e for e in validate_lines(bad))


def _eio_on(monkeypatch, name: str) -> None:
    """Reads of any file called `name` fail with EIO, as on a flaky bucket mount."""
    import errno

    real_read, real_copy = Path.read_bytes, shutil.copyfile

    def read_bytes(self):
        if self.name == name:
            raise OSError(errno.EIO, "Input/output error", str(self))
        return real_read(self)

    def copyfile(src, dst, *a, **kw):
        if Path(src).name == name:
            raise OSError(errno.EIO, "Input/output error", str(src))
        return real_copy(src, dst, *a, **kw)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(export.shutil, "copyfile", copyfile)


def _a_request_blob(run_dir: Path) -> str:
    return next(e["request_hash"] for e in _events(run_dir) if e["type"] == "inference_attempt")


def test_raw_copy_io_error_fails_the_export(llm_run, tmp_path, monkeypatch):
    sha = _a_request_blob(llm_run.dir)
    _eio_on(monkeypatch, sha)
    out = tmp_path / "e"
    with pytest.raises(export.ExportError, match=r"could not copy \d+ file.*Errno 5.*--no-raw"):
        export.export_run(llm_run.dir, out)
    assert not (out / "run.json").exists() and not export.is_current(llm_run.dir, out)
    res = CliRunner().invoke(app, ["export", str(llm_run.dir), "--out", str(out)])
    assert res.exit_code == 1 and "ExportError" in res.output and sha[:12] in res.output


def test_no_raw_export_reports_unreadable_blobs(llm_run, tmp_path, monkeypatch):
    out = export.export_run(llm_run.dir, tmp_path / "plain", raw=False)
    doc = json.loads((out / "run.json").read_text())
    assert doc["export_raw"] is False and doc["blobs"]["included"] == "none"
    assert doc["unreadable_blobs"] == {} and not (out / "raw").exists()
    assert export.is_current(llm_run.dir, out, raw=False) and not export.is_current(llm_run.dir, out)
    sha = _a_request_blob(llm_run.dir)
    _eio_on(monkeypatch, sha)
    res = CliRunner().invoke(app, ["export", str(llm_run.dir), "--out", str(tmp_path / "e"),
                                   "--no-raw"])
    assert res.exit_code == 0, res.output
    assert "warning: 1 blob(s) could not be read" in res.output
    doc = json.loads((tmp_path / "e" / "run.json").read_text())
    assert list(doc["unreadable_blobs"]) == [sha] and "Errno 5" in doc["unreadable_blobs"][sha]
    for f in sorted((tmp_path / "e" / "sessions").glob("*.jsonl")):
        assert validate_file(f) == [], f
