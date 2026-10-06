import hashlib

import pytest

from swarmlab.blobs import BlobStore


def test_put_get_has(tmp_path):
    bs = BlobStore(tmp_path / "blobs")
    sha = bs.put(b"hello")
    assert sha == hashlib.sha256(b"hello").hexdigest()
    assert bs.has(sha) and sha in bs
    assert bs.get(sha) == b"hello"
    assert (tmp_path / "blobs" / sha[:2] / sha).read_bytes() == b"hello"


def test_idempotent_and_empty(tmp_path):
    bs = BlobStore(tmp_path)
    assert bs.put(b"x", sync=True) == bs.put(b"x")
    assert bs.get(bs.put(b"")) == b""


def test_missing_and_invalid(tmp_path):
    bs = BlobStore(tmp_path)
    missing = "0" * 64
    assert not bs.has(missing)
    with pytest.raises(KeyError):
        bs.get(missing)
    with pytest.raises(ValueError):
        bs.get("../etc/passwd")
    with pytest.raises(TypeError):
        bs.put("text")  # type: ignore[arg-type]


def test_text_helpers_and_no_temp_left(tmp_path):
    bs = BlobStore(tmp_path)
    sha = bs.put_text("héllo")
    assert bs.get_text(sha) == "héllo"
    assert not [p for p in tmp_path.rglob("*") if p.name.endswith(".tmp")]


def test_reopen_sees_blobs(tmp_path):
    sha = BlobStore(tmp_path).put(b"persist")
    assert BlobStore(tmp_path).get(sha) == b"persist"
