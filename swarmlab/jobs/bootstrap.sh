#!/usr/bin/env bash
# swarmlab HF Job bootstrap (docs/handoff/WP8.md). Runs INSIDE the job, started by the command
# `swarmlab job run` prints:  fetch the stage dir from the bucket, then `bash bootstrap.sh`.
#
# Every setting comes from the environment (set by swarmlab/jobs/launch.py with `-e`):
#   SWARMLAB_TAG          job tag, e.g. m2-vllm-qwen9b-1a2b3c4-20261006-120000
#   SWARMLAB_STAGE        hf://buckets/<ns>/<bucket>/jobs/<tag>  (wheel, spec.yaml, this script)
#   SWARMLAB_RUNS_REMOTE  hf://buckets/<ns>/<bucket>/runs/<tag>  (run dirs are synced here)
#   SWARMLAB_RUNS         space-separated "arm:seed" pairs, run in this order
#   SWARMLAB_GIT_COMMIT / SWARMLAB_GIT_DIRTY / SWARMLAB_JOB_SPEC_SHA256 / SWARMLAB_FLAVOR / SWARMLAB_IMAGE
#   SWARMLAB_SYNC_EVERY   seconds between background syncs of $SWARMLAB_JOB_TMP/runs (default 300)
#   VLLM_MODEL VLLM_VERSION VLLM_MAX_MODEL_LEN VLLM_GPU_MEMORY_UTILIZATION VLLM_FLAGS
#   VLLM_REVISION (optional)  VLLM_WAIT_S (default 1200: /health + /v1/models must answer by then)
#
# Layout: vLLM in the image's system Python (`uv pip install --system`, as in the textworld
# study: the server must not share a venv with swarmlab); swarmlab + the hf CLI in their own
# venv; runs on local disk (/tmp/runs by default: bucket mounts drop exec bits and refuse sqlite/sockets),
# mirrored to the bucket with `hf buckets sync` after every run and every SWARMLAB_SYNC_EVERY s.
# job.json (manifest), results.jsonl (one CLI summary per run), vllm.log and job-side logs land
# in SWARMLAB_STAGE. Exit codes: 0 all runs ran; 1 a run failed; 10 install; 11 vLLM install;
# 12 vLLM not ready; 13 endpoint probe failed; 143 SIGTERM (job timeout or cancel).
set -uo pipefail

T0=$(date +%s)
log() { echo "[swarmlab-job +$(( $(date +%s) - T0 ))s] $*"; }
: "${SWARMLAB_TAG:?}" "${SWARMLAB_STAGE:?}" "${SWARMLAB_RUNS_REMOTE:?}" "${SWARMLAB_RUNS:?}" "${VLLM_MODEL:?}"
ROOT="${SWARMLAB_JOB_ROOT:-/opt/swarmlab}"   # venv + work dir
JT="${SWARMLAB_JOB_TMP:-/tmp}"               # local-disk scratch: runs, logs, manifest
STAGE_DIR="${SWARMLAB_STAGE_DIR:-$ROOT/stage}"
VENV="$ROOT/venv"
WORK="$ROOT/work"
OUT="$JT/runs"
SYNC_EVERY="${SWARMLAB_SYNC_EVERY:-300}"
VLLM_WAIT_S="${VLLM_WAIT_S:-1200}"
VLLM_PORT="${VLLM_PORT:-8000}"   # the spec's providers override points at this port
BASE="http://127.0.0.1:$VLLM_PORT"
export HF_HOME="${HF_HOME:-$JT/hf}" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export SWARMLAB_OFFLINE=1   # never fetch the router catalog inside the job
STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
STATUS=starting RC=0 VLLM_PID="" SYNC_PID="" RUN_PID="" READY_S="" FAILED_RUNS="" DONE_RUNS=""
HF="hf"
mkdir -p "$OUT" "$WORK" "$JT/joblogs"

hf_q() { "$HF" "$@" >>"$JT/joblogs/hf.log" 2>&1; }

# ---- manifest ---------------------------------------------------------------------------------
write_manifest() {
  ENDED_AT="${1:-}" STARTED_AT="$STARTED_AT" STATUS="$STATUS" RC="$RC" READY_S="$READY_S" \
    DONE_RUNS="$DONE_RUNS" FAILED_RUNS="$FAILED_RUNS" python3 - <<'PY' >"$JT/job.json"
import json, os
e = os.environ.get
doc = {
    "tag": e("SWARMLAB_TAG"), "job_id": e("JOB_ID"), "flavor": e("SWARMLAB_FLAVOR"),
    "accelerator": e("ACCELERATOR"), "image": e("SWARMLAB_IMAGE"),
    "commit": e("SWARMLAB_GIT_COMMIT"), "dirty": e("SWARMLAB_GIT_DIRTY") == "1",
    "job_spec_sha256": e("SWARMLAB_JOB_SPEC_SHA256"),
    "started_at": e("STARTED_AT"), "ended_at": e("ENDED_AT") or None,
    "status": e("STATUS"), "exit_code": int(e("RC") or 0),
    "runs_planned": (e("SWARMLAB_RUNS") or "").split(),
    "runs_done": (e("DONE_RUNS") or "").split(), "runs_failed": (e("FAILED_RUNS") or "").split(),
    "runs_remote": e("SWARMLAB_RUNS_REMOTE"),
    "vllm": {"model": e("VLLM_MODEL"), "revision": e("VLLM_REVISION") or None,
             "version": e("VLLM_VERSION"), "max_model_len": e("VLLM_MAX_MODEL_LEN"),
             "gpu_memory_utilization": e("VLLM_GPU_MEMORY_UTILIZATION"),
             "flags": e("VLLM_FLAGS"), "ready_s": e("READY_S") or None},
}
print(json.dumps(doc, indent=1))
PY
  hf_q buckets cp "$JT/job.json" "$SWARMLAB_STAGE/job.json" || log "WARNING: manifest upload failed"
}

# ---- bucket sync ------------------------------------------------------------------------------
sync_runs() {
  [ -d "$OUT" ] || return 0
  if command -v flock >/dev/null; then
    flock "$JT/swarmlab-sync.lock" "$HF" buckets sync "$OUT" "$SWARMLAB_RUNS_REMOTE" >>"$JT/joblogs/sync.log" 2>&1
  else
    "$HF" buckets sync "$OUT" "$SWARMLAB_RUNS_REMOTE" >>"$JT/joblogs/sync.log" 2>&1
  fi || log "WARNING: bucket sync failed (tail: $(tail -2 "$JT/joblogs/sync.log" | tr '\n' ' '))"
}

finish() {
  trap - EXIT TERM INT
  [ -n "$RUN_PID" ] && kill -TERM "$RUN_PID" 2>/dev/null && wait "$RUN_PID" 2>/dev/null
  [ -n "$SYNC_PID" ] && kill "$SYNC_PID" 2>/dev/null
  if [ "$STATUS" != "completed" ]; then STATUS="failed"; fi
  [ "$RC" -eq 0 ] && [ "$STATUS" = "failed" ] && RC=1
  log "finishing: status=$STATUS rc=$RC; final sync of $OUT -> $SWARMLAB_RUNS_REMOTE"
  sync_runs
  if [ -n "$VLLM_PID" ]; then
    grep -c -i preempt "$JT/vllm.log" 2>/dev/null | sed 's/^/vLLM preemption lines: /'
    kill "$VLLM_PID" 2>/dev/null
  fi
  [ -f "$JT/vllm.log" ] && hf_q buckets cp "$JT/vllm.log" "$SWARMLAB_STAGE/vllm.log"
  [ -f "$JT/results.jsonl" ] && hf_q buckets cp "$JT/results.jsonl" "$SWARMLAB_STAGE/results.jsonl"
  hf_q buckets sync "$JT/joblogs" "$SWARMLAB_STAGE/logs"
  write_manifest "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  log "JOB_EXIT $RC"
  exit "$RC"
}
trap 'RC=$?; finish' EXIT
trap 'log "signal: stopping (job timeout or cancel)"; RC=143; finish' TERM INT
fail() { RC="$1"; shift; log "FATAL: $*"; exit "$RC"; }

log "tag=$SWARMLAB_TAG job=${JOB_ID:-?} commit=${SWARMLAB_GIT_COMMIT:-?} accelerator=${ACCELERATOR:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || log "no GPU visible"

# ---- swarmlab (wheel from the stage dir) + hf CLI in a venv ----------------------------------
WHEEL=$(ls "$STAGE_DIR"/swarmlab-*.whl 2>/dev/null | head -1)
[ -n "$WHEEL" ] || fail 10 "no swarmlab wheel in $STAGE_DIR"
log "installing $(basename "$WHEEL") + huggingface_hub into $VENV"
uv venv -q --python 3.12 "$VENV" || fail 10 "uv venv failed"
uv pip install -q --python "$VENV/bin/python" "$WHEEL" "huggingface_hub>=1.32" || fail 10 "pip install of the wheel failed"
HF="$VENV/bin/hf"
SWARMLAB="$VENV/bin/swarmlab"
"$SWARMLAB" validate "$STAGE_DIR/spec.yaml" >/dev/null || fail 10 "spec.yaml does not validate"
write_manifest

# ---- vLLM -------------------------------------------------------------------------------------
STATUS=serving
log "uv pip install --system vllm==$VLLM_VERSION"
uv pip install --system -q "vllm==$VLLM_VERSION" || fail 11 "vLLM install failed"
VLLM_BIN=$(command -v vllm) || fail 11 "vllm executable not on PATH after install"
REV_ARGS=()
[ -n "${VLLM_REVISION:-}" ] && REV_ARGS=(--revision "$VLLM_REVISION")
log "starting: vllm serve $VLLM_MODEL --max-model-len $VLLM_MAX_MODEL_LEN --gpu-memory-utilization $VLLM_GPU_MEMORY_UTILIZATION $VLLM_FLAGS"
# shellcheck disable=SC2086  # VLLM_FLAGS is a flag list
"$VLLM_BIN" serve "$VLLM_MODEL" "${REV_ARGS[@]}" --served-model-name "$VLLM_MODEL" \
  --host 127.0.0.1 --port "$VLLM_PORT" --max-model-len "$VLLM_MAX_MODEL_LEN" \
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" $VLLM_FLAGS >"$JT/vllm.log" 2>&1 &
VLLM_PID=$!
WAIT_T0=$(date +%s)
while :; do
  if curl -sf "$BASE/health" >/dev/null && curl -sf "$BASE/v1/models" | grep -q "\"$VLLM_MODEL\""; then
    break
  fi
  kill -0 "$VLLM_PID" 2>/dev/null || { tail -40 "$JT/vllm.log"; fail 12 "vLLM exited before it was ready"; }
  if [ $(( $(date +%s) - WAIT_T0 )) -ge "$VLLM_WAIT_S" ]; then
    tail -40 "$JT/vllm.log"; fail 12 "vLLM not ready after ${VLLM_WAIT_S}s (/health and /v1/models)"
  fi
  sleep 5
done
READY_S=$(( $(date +%s) - WAIT_T0 ))
log "vLLM ready in ${READY_S}s: $(grep -E 'KV cache|Maximum concurrency' "$JT/vllm.log" | tail -2 | tr '\n' ' ')"

# one real tool-calling completion, thinking off: proves the parsers before any run starts
VLLM_PORT="$VLLM_PORT" VLLM_MODEL="$VLLM_MODEL" python3 - <<'PY' || fail 13 "endpoint probe failed"
import json, os, time, urllib.request
body = {"model": os.environ["VLLM_MODEL"], "max_tokens": 256,
        "messages": [{"role": "user", "content": "Guess candidate B with the tool."}],
        "tools": [{"type": "function", "function": {"name": "guess", "description": "Guess the flag.",
                   "parameters": {"type": "object", "properties": {"candidate": {"type": "string"}},
                                  "required": ["candidate"]}}}],
        "tool_choice": "auto", "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % os.environ.get("VLLM_PORT", "8000"), data=json.dumps(body).encode(),
                             headers={"content-type": "application/json"})
t0 = time.time()
r = json.load(urllib.request.urlopen(req, timeout=300))
ch = r["choices"][0]
calls = ch["message"].get("tool_calls") or []
print("PROBE finish=%s tool_calls=%s usage=%s %.1fs" % (ch.get("finish_reason"), json.dumps(calls)[:200],
      r.get("usage"), time.time() - t0))
if not calls:
    print("PROBE WARNING: no parsed tool call (check --tool-call-parser); runs will continue")
PY

# ---- runs -------------------------------------------------------------------------------------
STATUS=running
write_manifest
( while sleep "$SYNC_EVERY"; do sync_runs; done ) &
SYNC_PID=$!
cd "$WORK" || fail 10 "no work dir"
for item in $SWARMLAB_RUNS; do
  arm="${item%:*}"; seed="${item##*:}"
  log "run arm=$arm seed=$seed"
  # in the background + `wait`, so a SIGTERM (job timeout) reaches the trap at once
  "$SWARMLAB" run "$STAGE_DIR/spec.yaml" --arm "$arm" --seed "$seed" --yes --out "$OUT" --json \
    >"$JT/joblogs/run-$arm-s$seed.json" 2> >(tee "$JT/joblogs/run-$arm-s$seed.err" >&2) &
  RUN_PID=$!
  wait "$RUN_PID"
  rc=$?
  RUN_PID=""
  cat "$JT/joblogs/run-$arm-s$seed.json" >>"$JT/results.jsonl"
  if [ "$rc" -eq 0 ]; then DONE_RUNS="$DONE_RUNS $item"; else FAILED_RUNS="$FAILED_RUNS $item"; log "run $item failed (exit $rc)"; fi
  sync_runs
  hf_q buckets cp "$JT/results.jsonl" "$SWARMLAB_STAGE/results.jsonl"
done
if [ -z "$FAILED_RUNS" ]; then STATUS=completed; RC=0; else STATUS=failed; RC=1; fi
exit "$RC"
