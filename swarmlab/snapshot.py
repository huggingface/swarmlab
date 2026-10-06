"""Per-round snapshot manifests (docs/INTERFACE.md §13).

A snapshot is a JSON manifest at `<run_dir>/snapshots/<round:06d>.json` that points at a log
position (`log_seq`, the seq of that round's `round_committed`) plus one blob per plugin in the
run's `BlobStore`. Plugins never see the manifest; they only implement `snapshot()`/`restore()`.

Decisions where the contract is silent:

- `SnapshotStore(run_dir, blobs=None)`: `blobs` defaults to `BlobStore(run_dir / "blobs")`.
- `write(manifest, plugin_blobs)` stores each blob with `sync=True`, then sets
  `manifest.plugins[key] = sha` for every key in `plugin_blobs` (mutating the passed manifest;
  pre-existing entries for other keys are kept), then writes the manifest atomically and fsynced.
  Manifest JSON is deterministic: sorted keys, 2-space indent, trailing newline.
- `latest(max_round)` returns the manifest with the largest round `<= max_round` (any round if
  None), or None. `list_rounds()` is ascending. `read(round)` loads one manifest.
- `load(manifest)` raises `KeyError` naming the missing blob if any blob is absent.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ._io import atomic_write_bytes
from .blobs import BlobStore

_NAME = re.compile(r"^(\d{6,})\.json$")


class SnapshotManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run: str
    round: int
    log_seq: int
    # "world" | "board" | "scheduler" | f"participant:{agent}" | f"metric:{name}" -> blob sha
    plugins: dict[str, str] = {}
    outcomes_prev: dict[str, list[dict]] = {}
    live: list[str] = []

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, indent=2) + "\n"


class SnapshotStore:
    def __init__(self, run_dir: Path | str, blobs: BlobStore | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.dir = self.run_dir / "snapshots"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.blobs = blobs if blobs is not None else BlobStore(self.run_dir / "blobs")

    def path(self, round: int) -> Path:
        return self.dir / f"{round:06d}.json"

    def write(self, manifest: SnapshotManifest, plugin_blobs: dict[str, bytes]) -> Path:
        for key in sorted(plugin_blobs):
            manifest.plugins[key] = self.blobs.put(plugin_blobs[key], sync=True)
        manifest.plugins = dict(sorted(manifest.plugins.items()))
        return atomic_write_bytes(self.path(manifest.round), manifest.to_json().encode(), sync=True)

    def list_rounds(self) -> list[int]:
        rounds = []
        for p in self.dir.iterdir():
            m = _NAME.match(p.name)
            if m:
                rounds.append(int(m.group(1)))
        return sorted(rounds)

    def read(self, round: int) -> SnapshotManifest:
        return SnapshotManifest.model_validate_json(self.path(round).read_bytes())

    def latest(self, max_round: int | None = None) -> SnapshotManifest | None:
        rounds = [r for r in self.list_rounds() if max_round is None or r <= max_round]
        return self.read(rounds[-1]) if rounds else None

    def load(self, manifest: SnapshotManifest) -> dict[str, bytes]:
        return {key: self.blobs.get(sha) for key, sha in manifest.plugins.items()}
