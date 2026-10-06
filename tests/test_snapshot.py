import pytest

from swarmlab.blobs import BlobStore
from swarmlab.snapshot import SnapshotManifest, SnapshotStore


def manifest(r: int) -> SnapshotManifest:
    return SnapshotManifest(
        run="e__s1", round=r, log_seq=10 * r, outcomes_prev={"a000": [{"accepted": True}]},
        live=["a000", "a001"],
    )


def test_write_load_latest(tmp_path):
    store = SnapshotStore(tmp_path)
    for r in (1, 2, 3):
        path = store.write(manifest(r), {"world": f"w{r}".encode(), "participant:a000": b"p"})
        assert path == tmp_path / "snapshots" / f"{r:06d}.json"
    assert store.list_rounds() == [1, 2, 3]
    latest = store.latest()
    assert latest.round == 3 and latest.log_seq == 30
    assert store.load(latest) == {"participant:a000": b"p", "world": b"w3"}
    assert store.latest(max_round=2).round == 2
    assert store.latest(max_round=0) is None
    assert store.read(1).outcomes_prev == {"a000": [{"accepted": True}]}
    # blobs land in the run's blob store, dedup'd
    assert BlobStore(tmp_path / "blobs").get(latest.plugins["world"]) == b"w3"


def test_write_fills_manifest_and_is_deterministic(tmp_path):
    a, b = SnapshotStore(tmp_path / "a"), SnapshotStore(tmp_path / "b")
    m1, m2 = manifest(4), manifest(4)
    pa = a.write(m1, {"world": b"w", "board": b"b"})
    pb = b.write(m2, {"board": b"b", "world": b"w"})
    assert set(m1.plugins) == {"world", "board"}
    assert pa.read_bytes() == pb.read_bytes()


def test_empty_store_and_shared_blobs(tmp_path):
    blobs = BlobStore(tmp_path / "shared")
    store = SnapshotStore(tmp_path / "run", blobs)
    assert store.latest() is None and store.list_rounds() == []
    m = manifest(1)
    store.write(m, {"world": b"zz"})
    assert blobs.has(m.plugins["world"])
    (tmp_path / "run" / "snapshots" / "junk.txt").write_text("ignored")
    assert store.list_rounds() == [1]


def test_load_missing_blob_raises(tmp_path):
    store = SnapshotStore(tmp_path)
    m = manifest(1)
    m.plugins["world"] = "f" * 64
    with pytest.raises(KeyError):
        store.load(m)
