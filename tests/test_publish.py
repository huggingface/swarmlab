"""M4 acceptance 2: publish to a dataset repo (mocked HfApi; real Hub behind SWARMLAB_HUB=1),
idempotent re-publish, and `fetch-published` restoring a run that `Run.load` replays."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub.utils import filter_repo_objects
from typer.testing import CliRunner

from swarmlab import Run, export
from swarmlab import publish as pub
from swarmlab.cli import app

from .helpers import llm_agent_experiment, logical
from .pi_session import validate_file


class FakeHub:
    """In-memory stand-in for the slice of `HfApi` that swarmlab.publish uses."""

    def __init__(self) -> None:
        self.repos: dict[str, dict] = {}
        self.commits: list[tuple[str, str, list[str]]] = []

    def whoami(self):
        return {"name": "tester"}

    def create_repo(self, repo_id, *, repo_type=None, private=None, exist_ok=False):
        assert repo_type == "dataset"
        if repo_id in self.repos:
            assert exist_ok
            return
        self.repos[repo_id] = {"private": bool(private), "files": {}}

    def update_repo_settings(self, repo_id, *, private=None, repo_type=None):
        if private is not None:
            self.repos[repo_id]["private"] = private

    def repo_info(self, repo_id, *, repo_type=None):
        return SimpleNamespace(private=self.repos[repo_id]["private"])

    def list_repo_tree(self, repo_id, *, recursive=False, repo_type=None):
        for path, data in sorted(self.repos[repo_id]["files"].items()):
            if path.endswith(".parquet"):  # LFS-tracked on the Hub
                yield SimpleNamespace(path=path, blob_id="x" * 40,
                                      lfs=SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()))
            else:
                yield SimpleNamespace(path=path, blob_id=pub.git_blob_sha1(data), lfs=None)

    def create_commit(self, repo_id, operations, *, commit_message, repo_type=None):
        paths = []
        for op in operations:
            src = op.path_or_fileobj
            data = src if isinstance(src, bytes) else Path(src).read_bytes()
            self.repos[repo_id]["files"][op.path_in_repo] = data
            paths.append(op.path_in_repo)
        self.commits.append((repo_id, commit_message, paths))

    def hf_hub_download(self, repo_id, filename, *, repo_type=None, local_dir=None):
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self.repos[repo_id]["files"][filename])
        return str(p)

    def snapshot_download(self, repo_id, *, repo_type=None, local_dir=None, allow_patterns=None):
        files = self.repos[repo_id]["files"]
        for path in filter_repo_objects(list(files), allow_patterns=allow_patterns):
            p = Path(local_dir) / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(files[path])
        return str(local_dir)


@pytest.fixture(scope="module")
def runs_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("runs")
    exp = llm_agent_experiment(3, name="pubexp")
    exp.run_all([1, 2], max_rounds=3, out=out)
    # an unfinished run: skipped
    d = out / "pubexp__s9"
    shutil.copytree(out / "pubexp__s1", d)
    meta = json.loads((d / "run.json").read_text())
    (d / "run.json").write_text(json.dumps({**meta, "status": "running"}))
    return out


@pytest.fixture
def runs(runs_dir, tmp_path) -> Path:
    """A private copy per test (publish writes <run>/export)."""
    shutil.copytree(runs_dir, tmp_path / "runs")
    return tmp_path / "runs"


def test_publish_layout_card_and_index(runs):
    hub = FakeHub()
    res = pub.publish(runs, api=hub)
    assert res["skipped"] == ["pubexp__s9 (status running)"]
    (r,) = res["repos"]
    assert r["repo"] == "tester/pubexp" and r["public"] is False
    repo = hub.repos["tester/pubexp"]
    assert repo["private"] is True
    files = repo["files"]
    for rid in ("pubexp__s1", "pubexp__s2"):
        for fam in export.FAMILIES:
            assert f"runs/{rid}/tables/{fam}.parquet" in files
        assert f"runs/{rid}/raw/events.jsonl" in files
        assert f"runs/{rid}/run.json" in files
        assert files[f"runs/{rid}/raw/events.jsonl"] == (runs / rid / "events.jsonl").read_bytes()
        assert sum(1 for p in files if p.startswith(f"runs/{rid}/sessions/")) == 3
    index = json.loads(files["index.json"])
    assert sorted(index["runs"]) == ["pubexp__s1", "pubexp__s2"]
    assert index["runs"]["pubexp__s1"]["rows"]["turns"] == 9
    card = files["README.md"].decode()
    assert "format:agent-traces" not in card
    assert "config_name: turns" in card and "runs/*/sessions/*.jsonl" in card
    assert "`pubexp__s2`" in card and "## Table schemas" in card and "fake:reader" in card
    assert len(hub.commits) == 1


def test_republish_is_idempotent_and_public_adds_the_tag(runs):
    hub = FakeHub()
    pub.publish(runs, api=hub)
    again = pub.publish(runs, api=hub)["repos"][0]
    assert again["uploaded"] == 0 and len(hub.commits) == 1
    public = pub.publish(runs, api=hub, public=True, tag=["flaggame"])["repos"][0]
    assert public["public"] is True and hub.repos["tester/pubexp"]["private"] is False
    assert hub.commits[-1][2] == ["README.md"]  # only the card changed
    card = hub.repos["tester/pubexp"]["files"]["README.md"].decode()
    assert "format:agent-traces" in card and "- flaggame" in card


def test_fetch_published_restores_a_replayable_run(runs, tmp_path):
    hub = FakeHub()
    pub.publish(runs / "pubexp__s1", repo="me/flags", api=hub)
    dest = pub.fetch_published("me/flags", "pubexp__s1", tmp_path / "fetched", api=hub)
    orig = Run(runs / "pubexp__s1")
    loaded = Run.load(dest)
    assert loaded.score == orig.score and loaded.metrics == orig.metrics
    assert logical(loaded) == logical(orig)
    assert (dest / "export" / "tables" / "turns.parquet").exists()
    assert not (dest / "export" / "raw").exists()
    for f in (dest / "export" / "sessions").glob("*.jsonl"):
        assert validate_file(f) == []
    with pytest.raises(pub.PublishError, match="not empty"):
        pub.fetch_published("me/flags", "pubexp__s1", tmp_path / "fetched", api=hub)
    with pytest.raises(pub.PublishError, match="no run"):
        pub.fetch_published("me/flags", "nope", tmp_path / "other", api=hub)


def test_fetch_with_partial_blobs_still_loads(runs, tmp_path, monkeypatch):
    monkeypatch.setattr(export, "FULL_BLOBS_LIMIT", 1)
    hub = FakeHub()
    pub.publish(runs / "pubexp__s2", repo="me/flags", api=hub)
    files = hub.repos["me/flags"]["files"]
    assert not any("/raw/blobs/cache/" in p for p in files)
    dest = pub.fetch_published("me/flags", "pubexp__s2", tmp_path / "f", api=hub)
    assert Run.load(dest).score == Run(runs / "pubexp__s2").score


def test_experiment_publish_and_several_experiments(runs, tmp_path):
    other = llm_agent_experiment(2, name="otherexp")
    other.run(seed=1, max_rounds=2, out=runs)
    hub = FakeHub()
    with pytest.raises(pub.PublishError, match="several|belong"):
        pub.publish(runs, repo="me/x", api=hub)
    res = llm_agent_experiment(3, name="pubexp").publish(runs, repo="me/x", api=hub)
    assert res["repos"][0]["runs"] == ["pubexp__s1", "pubexp__s2"]
    both = pub.publish(runs, api=hub)
    assert sorted(r["repo"] for r in both["repos"]) == ["tester/otherexp", "tester/pubexp"]


def test_cli_publish_view_and_fetch(runs, tmp_path, monkeypatch):
    hub = FakeHub()
    monkeypatch.setattr(pub, "_api", lambda api=None: hub)
    cli = CliRunner()
    res = cli.invoke(app, ["publish", str(runs), "--repo", "me/flags", "--json"])
    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert data["repos"][0]["repo"] == "me/flags" and data["skipped"]
    res = cli.invoke(app, ["view", str(runs / "pubexp__s1"), "--publish", "me/flags", "--json"])
    assert res.exit_code == 0, res.output
    assert json.loads(res.output)["published"].endswith("runs/pubexp__s1/view.html")
    files = hub.repos["me/flags"]["files"]
    assert files["runs/pubexp__s1/view.html"] == (runs / "pubexp__s1" / "view.html").read_bytes()
    assert "runs/pubexp__s1/view.html" in files["README.md"].decode()
    assert json.loads(files["index.json"])["runs"]["pubexp__s1"]["view"] is True
    res = cli.invoke(app, ["fetch-published", "me/flags", "pubexp__s1", "--out",
                           str(tmp_path / "back"), "--json"])
    assert res.exit_code == 0, res.output
    assert json.loads(res.output)["run_id"] == "pubexp__s1"
    assert (tmp_path / "back" / "pubexp__s1" / "view.html").exists()
    res = cli.invoke(app, ["publish", str(tmp_path / "missing")])
    assert res.exit_code == 2


@pytest.mark.skipif(os.environ.get("SWARMLAB_HUB") != "1", reason="real Hub: set SWARMLAB_HUB=1")
def test_real_hub_round_trip(runs, tmp_path):
    """Publishes to a private repo, re-publishes (no upload), fetches, replays, deletes the repo."""
    from huggingface_hub import HfApi

    api = HfApi()
    repo = os.environ.get("SWARMLAB_HUB_REPO", "cmpatino/swarmlab-test-publish")
    try:
        first = pub.publish(runs / "pubexp__s1", repo=repo, api=api)["repos"][0]
        assert first["public"] is False and first["uploaded"] > 0
        assert api.repo_info(repo, repo_type="dataset").private is True
        again = pub.publish(runs / "pubexp__s1", repo=repo, api=api)["repos"][0]
        assert again["uploaded"] == 0
        dest = pub.fetch_published(repo, "pubexp__s1", tmp_path / "fetched", api=api)
        assert Run.load(dest).score == Run(runs / "pubexp__s1").score
    finally:
        api.delete_repo(repo, repo_type="dataset", missing_ok=True)


def test_publish_failures_exit_non_zero_and_name_the_step(runs, monkeypatch):
    """Field notes item 8: an export error (EIO on a bucket mount), a repo that cannot be
    created or an upload that fails is printed and exits 1; nothing looks published."""
    import errno

    hub = FakeHub()
    monkeypatch.setattr(pub, "_api", lambda api=None: hub)
    cli = CliRunner()

    def eio(*a, **kw):
        raise OSError(errno.EIO, "Input/output error")

    with monkeypatch.context() as m:
        m.setattr(pub, "export_run", eio)
        res = cli.invoke(app, ["publish", str(runs), "--repo", "me/flags"])
    assert res.exit_code == 1
    assert "PublishFailed: export of pubexp__s1 failed: OSError: [Errno 5]" in res.output
    assert hub.repos == {}

    with monkeypatch.context() as m:
        m.setattr(hub, "create_repo", lambda *a, **kw: eio())
        res = cli.invoke(app, ["publish", str(runs), "--repo", "me/flags", "--json"])
    assert res.exit_code == 1 and json.loads(res.stdout.splitlines()[-1])["ok"] is False
    assert "creating or opening the dataset repo me/flags failed" in res.output

    with monkeypatch.context() as m:
        m.setattr(hub, "create_commit", lambda *a, **kw: eio())
        res = cli.invoke(app, ["publish", str(runs), "--repo", "me/flags"])
    assert res.exit_code == 1 and "uploading to me/flags failed" in res.output
    assert hub.repos["me/flags"]["files"] == {}


def test_publish_no_raw_uploads_tables_and_sessions_only(runs, monkeypatch):
    hub = FakeHub()
    monkeypatch.setattr(pub, "_api", lambda api=None: hub)
    res = CliRunner().invoke(app, ["publish", str(runs), "--repo", "me/flags", "--no-raw"])
    assert res.exit_code == 0, res.output
    files = hub.repos["me/flags"]["files"]
    assert "runs/pubexp__s1/tables/turns.parquet" in files
    assert "runs/pubexp__s1/sessions/a000.jsonl" in files
    assert not any("/raw/" in p for p in files)
    entry = json.loads(files["index.json"])["runs"]["pubexp__s1"]
    assert entry["blobs"] == "none"
    assert json.loads((runs / "pubexp__s1" / "export" / "run.json").read_text())["export_raw"] is False
    # a later full publish re-exports with raw copies
    res = CliRunner().invoke(app, ["publish", str(runs), "--repo", "me/flags"])
    assert res.exit_code == 0 and "runs/pubexp__s1/raw/events.jsonl" in hub.repos["me/flags"]["files"]
