# WP8 handoff: HF Jobs placement with a co-located vLLM server

Branch `hf-jobs`. `uv run pytest -q -W error` and `uv run ruff check swarmlab tests examples tools`
are clean. No HF Job was launched while building this; every `hf` call in the tests is mocked
and the bootstrap script is exercised end to end against stub `uv`, `hf` and `vllm` executables.

DESIGN.md §12 says "HF Jobs default with a local option; one job hosts a wave of runs and vLLM
when self-hosting". HF Jobs have no inbound networking, so the vLLM server and the swarmlab
runner live in the same job, and results leave the job through an HF bucket.

## What landed

| file | what |
|---|---|
| `swarmlab/jobs/launch.py` | `plan_job` (validate, build the job spec, tag, estimate), `hf_command`, `describe`, `stage`, `submit`, `build_wheel`, `checkout_identity` |
| `swarmlab/jobs/bootstrap.sh` | runs inside the job: install, vLLM, readiness wait, endpoint probe, runs, bucket sync, manifest |
| `swarmlab/jobs/remote.py` | `status`, `logs`, `fetch`, `find_run` |
| `swarmlab/cli.py` | `swarmlab job run|status|logs|fetch`; run tables label self-hosted spend |
| `swarmlab/providers/openai_compat.py` | `self_hosted` kwarg (default: true iff name is `vllm`, left out of params when unset so older spec hashes hold); `served_by` falls back to `"vllm"` |
| `swarmlab/providers/base.py` | `Provider.self_hosted = False` |
| `swarmlab/experiment.py` | `Run.summary()["self_hosted"]` (prefixes, only when non-empty), `self_hosted_prefixes` |
| `swarmlab/spec.py` | `git_identity` honours `SWARMLAB_GIT_COMMIT` / `SWARMLAB_GIT_DIRTY` (set in the job) |
| `experiments/m2_vllm_qwen9b.yaml` | arms `bc-qwen9b-16`, `gossip-qwen9b-16`, `bc-qwen9b-64`, 10 rounds, seed 1, belief probe, world and probe-sourced belief metrics, zero budget |
| `tests/test_jobs.py` | 18 tests (command determinism and contents, job spec override, model-id validation, estimate, bootstrap pieces, bootstrap end to end with stubs incl. a failing run, stage, submit, CLI with and without `--launch`, fetch from a local stand-in and through a mocked `hf`, status) |
| `tools/m2_report.py` | ruff findings fixed (the lint gate covers `tools/`) |

## How it works

`swarmlab job run SPEC --model M` (prints only) and `... --launch` (submits):

1. **Plan.** Every model the chosen arms call (participants' `model`, probes' `coder_model`) must
   be `vllm:M`; scripted arms are allowed, an arm set that calls no model is refused, and any
   other prefix is refused (the job holds only `HF_TOKEN`). The **job spec** is the YAML plus
   ```yaml
   providers:
     vllm: {type: openai_compat, params: {name: vllm, base_url: "http://127.0.0.1:8000/v1",
            pricing: {"M": [0, 0, 0]}, concurrency: 64, timeout_s: 600.0, self_hosted: true}}
   ```
   and every chosen arm of it must build offline before anything is uploaded. The run specs
   recorded in the job therefore carry this provider: their `spec_hash` differs from a local
   run of the same YAML against another server, by design.
2. **Stage** (`<bucket>/jobs/<tag>/`): `spec.yaml` (job spec), `spec.orig.yaml`, `bootstrap.sh`,
   `plan.json`, and a wheel from `uv build --wheel` of this checkout. Tag =
   `<spec name>-<commit7>-<UTC yyyymmdd-HHMMSS>`. `--launch` refuses a dirty checkout unless
   `--allow-dirty` (the wheel would not match the recorded commit).
3. **Submit**: `hf buckets sync <stage> <bucket>/jobs/<tag>`, then `hf jobs run --detach ...`,
   parse the 24-hex job id, upload `launch.json`, append a row to `runs/jobs.jsonl` (`--ledger`).
4. **In the job** (`bootstrap.sh`): fetch the stage dir with `uv tool run --from
   'huggingface_hub>=1.32' hf buckets sync` (no bucket mount: mounts drop exec bits and refuse
   sqlite/sockets, and occasionally fail at start); venv with the wheel + `huggingface_hub`;
   `swarmlab validate spec.yaml`; `uv pip install --system vllm==0.30.0`; `vllm serve M
   --served-model-name M --host 127.0.0.1 --port 8000 --max-model-len 262144
   --gpu-memory-utilization 0.9 --dtype bfloat16 --language-model-only --reasoning-parser qwen3
   --enable-auto-tool-choice --tool-call-parser qwen3_coder` in the background; wait up to
   20 min for `/health` and `/v1/models` listing M (fails fast if the server process dies);
   one tool-calling completion with thinking off as a probe (warns if no tool call was parsed);
   then for each `arm:seed`: `swarmlab run spec.yaml --arm A --seed S --yes --out /tmp/runs
   --json`, `hf buckets sync /tmp/runs <bucket>/runs/<tag>` after every run and every 300 s in
   the background (the whole run dir, blobs and inference cache included, so replay and fork
   work locally). A failed run does not stop the others.
5. **Manifest** `<bucket>/jobs/<tag>/job.json` is written at install, when vLLM is ready, and at
   exit: tag, job id (`$JOB_ID`), flavor, accelerator, image, commit, dirty, job spec sha256,
   start/end, status (`starting|serving|running|completed|failed`), exit code, runs
   planned/done/failed, vLLM model, version, flags, max_model_len, gpu util, seconds to ready.
   Also uploaded: `results.jsonl` (one `swarmlab run --json` summary per run, per-run
   `spec_hash` included), `vllm.log`, `logs/` (per-run stderr, sync and hf logs).
6. **Exit codes**: 0 all runs ran; 1 a run failed; 10 install; 11 vLLM install; 12 vLLM not
   ready; 13 endpoint probe failed; 143 SIGTERM (timeout or cancel; the trap stops the current
   run, syncs, writes the manifest; the interrupted run is resumable from its last commit after
   `job fetch` + `swarmlab resume` against a server).

Then: `swarmlab job status JOB_ID` (stage, running seconds, compute $ so far),
`swarmlab job logs JOB_ID --follow`, `swarmlab job fetch RUN_ID [--tag TAG] [--out runs]`
(newest tag holding the run by default; `TAG/RUN_ID` also accepted), then `swarmlab view`,
`swarmlab replay`, `Run.load` locally. `swarmlab fork` works for the cached prefix; continuing
past it needs a vLLM server again, since the recorded base_url is the job's localhost (pass an
edited spec with your own `providers.vllm`).

Self-hosted spend: the ledger counts calls and tokens at price 0; `Run.summary()["self_hosted"]`
is `["vllm"]` and the `swarmlab run` table prints `$0.0000 +compute (self-hosted)`; the human
summary prints `self_hosted ['vllm']`. The real cost is the job's minutes on `job status`.

## Choices and why

- **Wheel, not `git+https`.** The repo is private; a git install needs a GitHub token as a job
  secret. The wheel needs only the HF token the job already uses for the bucket, and pins the
  exact code. The commit reaches `run.json` through `SWARMLAB_GIT_COMMIT`.
- **Image `ghcr.io/astral-sh/uv:python3.12-bookworm` + `vllm==0.30.0` via `uv pip install
  --system`.** This is the combination the textworld study ran Qwen/Qwen3.5-9B with on
  a100-large (`open-weights-coop/textworld/harness/jobs/batch_job.sh`, budget.json `vllm`):
  install 38 s, server up in ~5 min, KV cache 1.65M tokens, `qwen3_coder` parsed tool calls
  (147 requests, 146 `tool_calls`). vLLM stays in the system Python and swarmlab in its own venv
  (textworld's first probe failed by mixing the two). `vllm/vllm-openai:latest` is the
  alternative (`--image`), but it has no uv and an unpinned vLLM: untested here.
- **Flags.** `--dtype bfloat16 --language-model-only --reasoning-parser qwen3
  --enable-auto-tool-choice --tool-call-parser qwen3_coder` from textworld, plus the requested
  `--gpu-memory-utilization 0.9` (textworld used 0.92) and `--max-model-len 262144` (full
  context: an over-long prompt is an HTTP 400, which fails the run, so keep it large; full
  memory at N=64 broadcast grows the prompt every round). `VLLM_USE_FLASHINFER_SAMPLER=0` as in
  textworld. `--revision` pins the model (textworld pinned
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`); not pinned by default.
- **Provider override.** concurrency 64 (N=64 agents call at once; vLLM batches them), per-request
  timeout 600 s (a 2048-token completion under a full batch can exceed the 90 s default), price 0.
- **One job, runs in sequence.** Arms and seeds run one after the other against one server
  (phase-commit already runs the N agents of a round concurrently). Running arms in parallel
  would cut wall time further; not needed at this size.
- **Estimate.** setup 600 s + per run `rounds x (30 + 2N)` s: a guess (router-served Qwen phase-1
  runs took ~1 min/round at N=16; a local server should not be slower). Default timeout = 1.5x
  the estimate rounded up to 10 min. Override with `--per-round` / `--timeout`.

## The approved-command shape

What to run first (prints; add `--launch` after the checklist below):

```
swarmlab job run experiments/m2_vllm_qwen9b.yaml --model Qwen/Qwen3.5-9B --flavor a100-large --timeout 2h
```

It expands to (tag and commit vary; checked with `hf jobs run --dry-run`, which parses it and
submits nothing):

```
hf jobs run --detach --flavor a100-large --timeout 2h --secrets HF_TOKEN \
  --name swarmlab-<tag> --label experiment=swarmlab --label swarmlab_tag=<tag> \
  -e SWARMLAB_TAG=<tag> \
  -e SWARMLAB_STAGE=hf://buckets/cmpatino/swarmlab-runs/jobs/<tag> \
  -e SWARMLAB_RUNS_REMOTE=hf://buckets/cmpatino/swarmlab-runs/runs/<tag> \
  -e 'SWARMLAB_RUNS=bc-qwen9b-16:1 gossip-qwen9b-16:1 bc-qwen9b-64:1' \
  -e SWARMLAB_GIT_COMMIT=<sha> -e SWARMLAB_GIT_DIRTY=0 -e SWARMLAB_JOB_SPEC_SHA256=<sha256> \
  -e SWARMLAB_FLAVOR=a100-large -e SWARMLAB_IMAGE=ghcr.io/astral-sh/uv:python3.12-bookworm \
  -e SWARMLAB_SYNC_EVERY=300 -e VLLM_MODEL=Qwen/Qwen3.5-9B -e VLLM_VERSION=0.30.0 \
  -e VLLM_MAX_MODEL_LEN=262144 -e VLLM_GPU_MEMORY_UTILIZATION=0.9 \
  -e 'VLLM_FLAGS=--dtype bfloat16 --language-model-only --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder' \
  -e VLLM_WAIT_S=1200 -e HF_HOME=/tmp/hf \
  ghcr.io/astral-sh/uv:python3.12-bookworm \
  bash -c 'mkdir -p /opt/swarmlab/stage && uv tool run --from '"'"'huggingface_hub>=1.32'"'"' hf buckets sync "$SWARMLAB_STAGE" /opt/swarmlab/stage && exec bash /opt/swarmlab/stage/bootstrap.sh'
```

Printed estimate for this spec: ~57 min wall (10 min setup; ~10 min per N=16 run, ~26 min for
N=64), timeout 2h. a100-large is $2.50/h, billed per minute while Starting or Running
(https://huggingface.co/docs/hub/jobs-pricing): ~$2.40 expected, $5.00 worst case. Without
`--timeout` the default would be 90m (1.5x the estimate); 2h leaves room for a slow N=64 run.
The default HF Jobs timeout is 30 minutes, so always pass one (the CLI always does).

Flavor notes (`hf jobs hardware`, 2026-10-06): a 9B bf16 model (~18 GB weights) fits a single
`l40sx1` (48 GB, $1.80/h) or `a10g-large` (24 GB, $1.50/h, too tight for a useful KV cache at
long context). `a100-large` (80 GB, $2.50/h) is the proven one. `h200` ($5/h) only if N=64
turns out KV-bound (watch `Running:`/preemption lines in `vllm.log`).

## First-launch checklist

1. The bucket does not exist yet (`hf buckets info cmpatino/swarmlab-runs` -> not found on
   2026-10-06). Create it, private: `hf buckets create cmpatino/swarmlab-runs --private`
   (ttc-qwen and textworld used `hf://buckets/cmpatino/<study>` the same way). A dataset repo is
   not needed: the job syncs with `hf buckets sync`.
2. `HF_TOKEN` in the launching shell has write access to the bucket and can read
   Qwen/Qwen3.5-9B (passed as `--secrets HF_TOKEN`; the job uses it for the stage download,
   the weights and the result sync).
3. Commit and push first: `--launch` refuses a dirty checkout; the wheel is built from the
   working tree.
4. Run without `--launch`, read the command, the run list and the estimate; get operator
   approval for the flavor and timeout; then the same command with `--launch`.
5. Watch: `swarmlab job logs <id> --follow` should show `vLLM ready in ...s`, a `PROBE
   finish=tool_calls` line, then `run arm=... seed=...` lines. If the probe prints `PROBE
   WARNING: no parsed tool call`, cancel (`hf jobs cancel <id>`): the tool parser is wrong and
   every agent turn would be prose.
6. After it ends: `swarmlab job status <id>` (compute cost), then `swarmlab job fetch
   m2-vllm-qwen9b__bc-qwen9b-16__s1` etc., `swarmlab view runs/<run_id>`.

## Open risks

- **Not run on real hardware.** The bootstrap was exercised with stubs; the first launch is the
  real test of `uv tool run ... hf buckets sync` as the entrypoint, of `vllm serve` from the
  system install and of the readiness check. `hf jobs run --dry-run` accepted the command.
- **vLLM version / image.** `vllm==0.30.0` on the uv bookworm image worked for textworld on
  2026-09-30; a newer wheel or a CUDA driver change could break it. The tag is pinned for that
  reason; `--vllm-version` and `--image` override.
- **Tool parser name.** `qwen3_coder` parsed Qwen3.5-9B tool calls in textworld (with thinking
  on). swarmlab sends `chat_template_kwargs: {enable_thinking: false}`; the endpoint probe uses
  the same setting and logs whether a tool call was parsed.
- **Context and memory at N=64.** Full-memory broadcast at N=64 grows prompts each round;
  64 concurrent long prompts may exceed the KV cache and cause preemption (slower, not wrong).
  The per-request timeout is 600 s with 2 retries.
- **Bucket sync of a live run dir** uploads partially written files; the next sync and the final
  one fix them. If the job is killed hard (no SIGTERM grace), the last <=5 min are lost but the
  run stays resumable from what was synced.
- **Fork past the cache** needs a server: the recorded provider points at the job's localhost.
