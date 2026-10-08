# Analysing and publishing results

Reports, Parquet exports, analysis patterns and publishing to the Hub. Back to the [docs index](../README.md).

## Analysing results

- `swarmlab prompts SPEC --arm A` prints the system prompt and round-1 user message of one agent per participant group, exactly as the model would receive them, without calling a model. Review them (ideally with a second agent) before spending.
- `swarmlab report RUNS_DIR` writes a Markdown report over the finished runs: per-arm summary, the trajectory of every logged metric, reading behaviour and a protocol-health section for any world (turn end kinds, errored turns with the first error, finish reasons, retries, latency, calls per turn, cost per round, rejected tool calls and refused world actions, skipped probes); Flag Game sections (where the swarm went, terminal states as in the Flag Game paper, probe vs world belief) and a coloring section appear only for those worlds. Simulated runs (only `fake:` models, e.g. dry runs) are listed but left out of the tables and spend totals; `--include-fake` adds them as `<arm> (simulated)`. `--out report.md` writes the file and prints one line (`--stdout` also prints the report).
- `swarmlab export RUN_DIR` (Python `Run.export(out)`) writes `RUN_DIR/export/`: `run.json` (identity, spec, score, spend, `spend_discarded_usd`, metric finals), `tables/<family>.parquet` (turns, tool_calls, posts, deliveries, reads, actions, inference, probes, metrics, interventions, rounds, run, other, and `discarded_inference`: the calls of rounds discarded at a hard ceiling, with `charged_usd`, so the ledger spend = non-cached `inference.cost_usd` + discarded spend; key columns `experiment, arm, seed, run, round, agent`; blob content inlined up to 64 KiB), `sessions/<agent>.jsonl` (one [pi-format](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/session-format.md) session per agent with `harness: "swarmlab"`, so the Hub's agent-traces viewer renders each conversation) and `raw/` (the byte-identical log, snapshots and blobs).

### Analysis patterns

- **Metrics are per round**: the runner feeds a metric every logged event and logs `value()` once per committed round. A per-decision series (e.g. under `commit: immediate`) comes from the `actions` table, not from metrics.
- **Put per-decision state you will analyse into `Outcome.feedback`**: it is logged in `action_committed` and lands, as JSON, in the `actions` table's `feedback` column next to `round`, `agent` and `action_args`. Agents see it too, so it must not reveal correctness.
- **Agents' reasoning text** is in the pi-format sessions (`export/sessions/<agent>.jsonl`, one `assistant` message per model call with its text and tool calls); that is the intended source, not the event log.
- **A metric that needs the truth** (what only `World.verify()` knows) returns `needs_truth() -> True`; the runner then calls `set_truth(world.verify())` before the first round. Mirrors `swarmlab/metrics/coloring.py`; use it as `metrics: ["mymetrics:BestSiteShare"]`:

```python
from swarmlab.metrics.base import Metric

class BestSiteShare(Metric):
    name = "sites.best_share"             # share of accepted picks on the best site
    def __init__(self):
        self.best, self.hits, self.n = None, 0, 0
    def needs_truth(self):
        return True                        # the runner calls set_truth(world.verify())
    def set_truth(self, truth):
        self.best = truth["best_site"]
    def update(self, event):               # every logged event, in log order
        if event.type == "action_committed" and event.accepted:
            self.n += 1
            self.hits += event.feedback.get("site") == self.best
    def value(self):                       # once per round: (value, denominator)
        return (self.hits / self.n if self.n else None), self.n
```

## Publishing

- `swarmlab publish runs/` (Python `Experiment.publish(runs_dir)`) exports what is needed and uploads every finished run to one **private** Hub dataset repo per experiment (`<you>/<experiment>`, or `--repo`): `runs/<run_id>/...`, `index.json`, and a dataset card with the arms' specs, a run table and the table schemas. Re-publishing uploads only changed files. A failed step (export, repo creation, upload) is printed and exits 1. `--public` makes the repo public and adds the `format:agent-traces` tag (the Hub's trace viewer renders public repos only); traces hold every prompt and reply, so read them first. Needs the `hub` extra and `HF_TOKEN`.
- Raw copies read every blob, which is slow on a bucket-mounted runs dir (thousands of small files) and can hit `[Errno 5]`; an unreadable file fails the export rather than being skipped. There, `export --no-raw` / `publish --no-raw` write and upload tables and sessions only (`run.json` `export_raw: false`; `fetch-published` cannot rebuild such a run), or publish from a local copy of the runs dir.
- `swarmlab view RUN_DIR --publish OWNER/REPO` uploads `view.html` next to the run and links it from the card.
- `swarmlab fetch-published OWNER/REPO RUN_ID --out runs/` rebuilds the run directory from the published raw log, snapshots and blobs, so `replay`, `view`, `fork` and `Run.load` work on it.
